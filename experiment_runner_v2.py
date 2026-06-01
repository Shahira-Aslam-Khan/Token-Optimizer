"""
experiment_runner.py
────────────────────
Measures real token and cost reduction from the TokenOptimizer library.

WHY THE OLD EXPERIMENT DIDN'T SHOW CACHE HITS
──────────────────────────────────────────────
The previous version ran each unique task exactly once. A cache needs
*repeated or similar queries* to show value. Running 10 unique tasks
once each gives 0% hit rate by design, not because the cache is broken.

HOW THIS EXPERIMENT IS STRUCTURED
──────────────────────────────────
We simulate a realistic production workload with three properties:

  1. Task repetition    — the same task appears multiple times
                          (users retry, pipelines re-run, batch jobs repeat)

  2. Semantic variation — the same underlying task is asked in slightly
                          different words (different users, paraphrases)

  3. Unique tasks       — some tasks are truly novel (cold cache misses)

Each base task is expanded into 4 requests:
  - original       → cold miss, response stored in cache
  - paraphrase_1   → should hit semantic cache
  - paraphrase_2   → should hit semantic cache
  - exact_repeat   → should hit exact cache

This mirrors how a real coding assistant, batch eval pipeline, or
code-gen service behaves: same problems, worded differently.

THREE CONDITIONS COMPARED
──────────────────────────
  baseline      — raw unoptimized prompt, no library
  caveman       — OutputLayer only: replace verbose system prompt with tight
                  brevity-fused version (build_system_prompt). No other
                  compression. Isolates the ~50-token gain from BrevityMode
                  before any PreCallLayer or CacheLayer work runs.
  pre_call_only — PreCallLayer only (compression + pruning + BrevityMode, no cache)
  full_v2       — Full pipeline: OutputLayer + PreCallLayer + SemanticResponseCache

Usage
─────
  python experiment_runner.py                        # dry-run, mock model
  python experiment_runner.py --real                 # real APIs
  python experiment_runner.py --n-base-tasks 10
  python experiment_runner.py --models gpt-4o-mini claude-haiku
"""

import os
import sys
import time
import json
import random
import argparse
import traceback
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from token_optimizer_v2 import (
    TokenOptimizerV2,
    TokenOptimizer,
    build_system_prompt,
    BREVITY_CODING_INSTRUCTIONS,
    BREVITY_POSITIONS,
)
from dataset_loader import DatasetLoader, CodeTask


# ──────────────────────────────────────────────────────────────────────────────
# SHARED PROMPT INPUTS
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Python software engineer.
Please note that you should note that you should always make sure to write clean, correct, idiomatic Python code.
It is important that you make sure to handle edge cases.
You should note that you should write well-structured code.
As a helpful assistant, please note that you should ensure correctness.
Return only the function code, no explanations, no markdown, no preamble."""

FEW_SHOT_EXAMPLES = [
    ("Write a function `add(a, b)` that returns the sum.",
     "def add(a, b):\n    return a + b"),
    ("Write a function `multiply(a, b)` that returns the product.",
     "def multiply(a, b):\n    return a * b"),
    ("Write a function `greet(name)` that returns 'Hello, name!'.",
     "def greet(name):\n    return f'Hello, {name}!'"),
    ("Write a function `square(n)` that returns n squared.",
     "def square(n):\n    return n * n"),
]


# ──────────────────────────────────────────────────────────────────────────────
# WORKLOAD BUILDER
# Expands base tasks into a realistic request sequence
# ──────────────────────────────────────────────────────────────────────────────

def _fn_name(prompt: str) -> str:
    import re
    m = re.search(r'`(\w+)`', prompt)
    return m.group(1) if m else "solution"


def build_workload(base_tasks: List[CodeTask]) -> List[Tuple[CodeTask, str]]:
    """
    Expands each base task into 4 requests:
      original     — exact as written (cold miss)
      paraphrase_1 — light reword     (semantic hit expected)
      paraphrase_2 — terser reword    (semantic hit expected)
      exact_repeat — identical copy   (exact hit expected)

    Shuffled with a fixed seed for reproducibility.
    """
    workload: List[Tuple[CodeTask, str]] = []

    for task in base_tasks:
        fn = _fn_name(task.prompt)

        # paraphrase 1: swap "Write" → "Implement", light verb changes
        p1 = deepcopy(task)
        p1.task_id = task.task_id + "-P1"
        p1.prompt = (task.prompt
                     .replace("Write a Python function", "Implement a Python function")
                     .replace("Write a function", "Create a function"))

        # paraphrase 2: strip to essentials — first sentence + function name
        import re
        first_line = task.prompt.strip().split("\n")[0].strip()
        p2 = deepcopy(task)
        p2.task_id = task.task_id + "-P2"
        p2.prompt = f"Implement `{fn}`. {first_line}"

        # exact repeat
        rep = deepcopy(task)
        rep.task_id = task.task_id + "-REP"

        workload += [
            (task, "original"),
            (p1,   "paraphrase_1"),
            (p2,   "paraphrase_2"),
            (rep,  "exact_repeat"),
        ]

    random.Random(42).shuffle(workload)
    return workload


# ──────────────────────────────────────────────────────────────────────────────
# COST TABLE
# ──────────────────────────────────────────────────────────────────────────────

MODEL_COSTS = {
    "gpt-3.5-turbo": {"input": 0.0005,  "output": 0.0015},
    "gpt-4o-mini":   {"input": 0.00015, "output": 0.0006},
    "gpt-4o":        {"input": 0.005,   "output": 0.015},
    "claude-haiku":  {"input": 0.00025, "output": 0.00125},
    "claude-sonnet": {"input": 0.003,   "output": 0.015},
    "mock-model":    {"input": 0.001,   "output": 0.002},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    c = MODEL_COSTS.get(model, MODEL_COSTS["mock-model"])
    return (input_tokens / 1000 * c["input"]) + (output_tokens / 1000 * c["output"])


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT RECORD
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentRecord:
    task_id: str
    variant: str              # original | paraphrase_1 | paraphrase_2 | exact_repeat
    model: str
    condition: str            # baseline | pre_call_only | full_v2
    original_tokens: int
    input_tokens: int         # tokens sent to LLM (0 on cache hit)
    output_tokens: int
    tokens_saved_compression: int
    compression_pct: float
    cache_hit: bool
    cache_tier: Optional[str] # exact | semantic | None
    latency_ms: float
    estimated_cost_usd: float
    generated_code: str
    tests_passed: int
    tests_total: int
    pass_at_1: float


# ──────────────────────────────────────────────────────────────────────────────
# CODE EVALUATOR
# ──────────────────────────────────────────────────────────────────────────────

class CodeEvaluator:
    def evaluate(self, code: str, tests: List[str], entry_point: str) -> Tuple[int, int]:
        passed, total = 0, len(tests)
        try:
            ns: dict = {}
            exec(code, ns)
        except Exception:
            return 0, total
        for t in tests:
            try:
                exec(t, ns)
                passed += 1
            except Exception:
                pass
        return passed, total


# ──────────────────────────────────────────────────────────────────────────────
# LLM CLIENT
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str, openai_key: str = "",
                 anthropic_key: str = "", dry_run: bool = True):
        self.model = model
        self.dry_run = dry_run
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key

    def generate(self, prompt: str, entry_point: str = "") -> Tuple[str, int, int, float]:
        if self.dry_run:           return self._mock(prompt, entry_point)
        if self.model.startswith("gpt"):    return self._openai(prompt)
        if self.model.startswith("claude"): return self._anthropic(prompt)
        return self._mock(prompt, entry_point)

    def _mock(self, prompt: str, entry_point: str = "") -> Tuple[str, int, int, float]:
        import re
        input_tokens = max(1, len(prompt) // 4)
        time.sleep(0.01)
        if entry_point:
            fn = entry_point
        else:
            m = re.search(r'`(\w+)`|function (\w+)', prompt)
            fn = (m.group(1) or m.group(2)) if m else "solution"
        T = {
            "has_close_elements":     "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i+1,len(numbers)):\n            if abs(numbers[i]-numbers[j])<threshold: return True\n    return False",
            "separate_paren_groups":  "def separate_paren_groups(paren_string):\n    r,cur,d=[],'' ,0\n    for c in paren_string:\n        if c=='(': d+=1;cur+=c\n        elif c==')': d-=1;cur+=c\n        if d==0 and cur: r.append(cur);cur=''\n    return r",
            "truncate_number":        "def truncate_number(number):\n    return number % 1.0",
            "below_zero":             "def below_zero(ops):\n    b=0\n    for op in ops:\n        b+=op\n        if b<0: return True\n    return False",
            "mean_absolute_deviation":"def mean_absolute_deviation(numbers):\n    m=sum(numbers)/len(numbers)\n    return sum(abs(x-m) for x in numbers)/len(numbers)",
            "find_max_sum":           "def find_max_sum(arr):\n    mx=cur=arr[0]\n    for n in arr[1:]:\n        cur=max(n,cur+n);mx=max(mx,cur)\n    return mx",
            "is_palindrome":          "def is_palindrome(s):\n    c=''.join(ch.lower() for ch in s if ch.isalnum())\n    return c==c[::-1]",
            "flatten_list":           "def flatten_list(n):\n    r=[]\n    for i in n: r.extend(flatten_list(i)) if isinstance(i,list) else r.append(i)\n    return r",
            "count_words":            "def count_words(s):\n    c={}\n    for w in s.lower().split(): c[w]=c.get(w,0)+1\n    return c",
            "rotate_list":            "def rotate_list(lst,k):\n    if not lst: return lst\n    k=k%len(lst)\n    return lst[-k:]+lst[:-k] if k else lst[:]",
        }
        code = T.get(fn, f"def {fn}(*args,**kwargs):\n    pass")
        return code, input_tokens, max(1, len(code)//4), random.uniform(150, 600)

    def _openai(self, prompt: str) -> Tuple[str, int, int, float]:
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_key)
            print(f"  [OpenAI] API Key loaded: {self.openai_key[:20]}...")
            t = time.time()
            r = client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":"Return ONLY the Python function. No explanations."},
                          {"role":"user","content":prompt}],
                max_tokens=512, temperature=0.2,
            )
            print(f"  [OpenAI] ✓ Success: {r.usage.prompt_tokens} input, {r.usage.completion_tokens} output")
            return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens, (time.time()-t)*1000
        except Exception as e:
            print(f"  [OpenAI Error] {e}"); return self._mock(prompt)

    def _anthropic(self, prompt: str) -> Tuple[str, int, int, float]:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            print(f"  [Anthropic] API Key loaded: {self.anthropic_key[:20]}...")
            mid = "claude-haiku-4-5-20251001" if "haiku" in self.model else "claude-sonnet-4-6"
            t = time.time()
            r = client.messages.create(model=mid, max_tokens=512,
                system="Return ONLY the Python function. No explanations.",
                messages=[{"role":"user","content":prompt}])
            print(f"  [Anthropic] ✓ Success: {r.usage.input_tokens} input, {r.usage.output_tokens} output")
            return r.content[0].text, r.usage.input_tokens, r.usage.output_tokens, (time.time()-t)*1000
        except Exception as e:
            print(f"  [Anthropic Error] {e}"); return self._mock(prompt)


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ──────────────────────────────────────────────────────────────────────────────

CONDITIONS = ["baseline", "caveman", "pre_call_only", "full_v2"]


class ExperimentRunner:
    def __init__(self, models: List[str], dry_run: bool = True,
                 openai_key: str = "", anthropic_key: str = "",
                 brevity_position: str = "pre_precall"):
        self.models = models
        self.dry_run = dry_run
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key
        self.brevity_position = brevity_position
        self.evaluator = CodeEvaluator()
        self.records: List[ExperimentRecord] = []

        # Stateless v1 optimizer shared across tasks
        self.pre_call = TokenOptimizer(
            use_rule_based=True, use_semantic_dedup=True,
            use_few_shot_pruning=True, use_instruction_distill=True,
            few_shot_top_k=2,
        )

        # OutputLayer-only optimizer: replaces system prompt with brevity-fused
        # version, all other stages disabled. Isolates BrevityMode savings.
        self.caveman = TokenOptimizerV2(
            brevity_mode="full",
            brevity_position=brevity_position,
            use_rule_based=False,
            use_semantic_dedup=False,
            use_few_shot_pruning=False,
            use_instruction_distill=False,
            use_response_cache=False,
            use_prefix_injection=False,
        )

    def run(self, base_tasks: List[CodeTask]) -> List[ExperimentRecord]:
        workload = build_workload(base_tasks)
        total = len(workload) * len(self.models) * len(CONDITIONS)

        print(f"\n{'='*76}")
        print(f"  {len(base_tasks)} base tasks → {len(workload)} requests")
        print(f"  (original + paraphrase_1 + paraphrase_2 + exact_repeat per task)")
        print(f"  {len(workload)} requests × {len(self.models)} models × {len(CONDITIONS)} conditions = {total} runs")
        print(f"  Mode: {'DRY RUN (mock)' if self.dry_run else 'REAL API'}")
        print(f"  Brevity position: {self.brevity_position}")
        print(f"{'='*76}\n")
        print(f"  {'#':>3}  {'model':<14} {'condition':<16} {'task':<12} {'variant':<14} {'tok':>5}  {'cost':>9}  {'pass':>5}  {'cache'}")
        print(f"  {'─'*3}  {'─'*14} {'─'*16} {'─'*12} {'─'*14} {'─'*5}  {'─'*9}  {'─'*5}  {'─'*12}")

        n = 0
        for model in self.models:
            client = LLMClient(model, self.openai_key, self.anthropic_key, self.dry_run)

            # Fresh v2 per model — independent cache per model
            v2 = TokenOptimizerV2(
                brevity_mode="full",
                brevity_position=self.brevity_position,  # CLI-controlled
                use_rule_based=True, use_semantic_dedup=True,
                use_few_shot_pruning=True, use_instruction_distill=True,
                few_shot_top_k=2,
                use_response_cache=True,
                use_prefix_injection=True,
                similarity_threshold=0.90,
                min_prefix_tokens=80,
                cost_per_1k_tokens=MODEL_COSTS.get(model, MODEL_COSTS["mock-model"])["input"],
            )

            for task, variant in workload:
                n += 1
                for condition in CONDITIONS:
                    try:
                        rec = self._run_one(client, v2, task, variant, condition, model)
                        self.records.append(rec)
                        hit_str = f"✓ {rec.cache_tier}" if rec.cache_hit else "—"
                        print(
                            f"  {n:>3}  {model:<14} {condition:<16} {task.task_id:<12} "
                            f"{variant:<14} {rec.input_tokens:>5}  "
                            f"${rec.estimated_cost_usd:.5f}  "
                            f"{rec.pass_at_1:>5.0%}  {hit_str}"
                        )
                    except Exception as e:
                        print(f"  ERROR [{condition}] {task.task_id}: {e}")
                        traceback.print_exc()

        return self.records

    def _run_one(self, client, v2, task, variant, condition, model):
        original_tokens = max(1, (
            len(SYSTEM_PROMPT) +
            sum(len(i)+len(o) for i, o in FEW_SHOT_EXAMPLES) +
            len(task.prompt)
        ) // 4)

        if condition == "baseline":
            parts = [SYSTEM_PROMPT]
            parts += [f"Input: {i}\nOutput: {o}" for i, o in FEW_SHOT_EXAMPLES]
            parts.append(f"Task: {task.prompt}")
            prompt = "\n\n".join(parts)
            gen, in_tok, out_tok, ms = client.generate(prompt, task.entry_point)
            return self._make(task, variant, model, condition, gen,
                              original_tokens, in_tok, out_tok, ms, 0, 0.0, False, None)

        if condition == "caveman":
            # OutputLayer only: swap verbose system prompt → brevity-fused version.
            # Examples and task are passed through raw — no compression, no cache.
            tight_system = build_system_prompt("full")
            parts = [tight_system]
            parts += [f"Input: {i}\nOutput: {o}" for i, o in FEW_SHOT_EXAMPLES]
            parts.append(f"Task: {task.prompt}")
            prompt = "\n\n".join(parts)
            gen, in_tok, out_tok, ms = client.generate(prompt, task.entry_point)
            saved = max(0, original_tokens - in_tok)
            return self._make(task, variant, model, condition, gen,
                              original_tokens, in_tok, out_tok, ms,
                              saved, round(saved / original_tokens * 100, 1), False, None)

        if condition == "pre_call_only":
            opt = self.pre_call.optimize(
                system_prompt=SYSTEM_PROMPT, user_prompt=task.prompt,
                context="", examples=FEW_SHOT_EXAMPLES, query=task.prompt,
            )
            gen, in_tok, out_tok, ms = client.generate(opt.optimized_prompt, task.entry_point)
            saved = max(0, original_tokens - in_tok)
            return self._make(task, variant, model, condition, gen,
                              original_tokens, in_tok, out_tok, ms,
                              saved, round(saved/original_tokens*100, 1), False, None)

        # full_v2
        prep = v2.prepare(system_prompt=SYSTEM_PROMPT, context="",
                          examples=FEW_SHOT_EXAMPLES, task=task.prompt)
        if prep.cache_hit:
            gen = prep.response
            in_tok = out_tok = 0
            ms = 0.0
        else:
            gen, in_tok, out_tok, ms = client.generate(prep.prompt, task.entry_point)
            v2.record(prep, gen, tokens_used=in_tok)

        return self._make(task, variant, model, condition, gen,
                          prep.original_tokens, in_tok, out_tok, ms,
                          prep.tokens_saved_by_compression,
                          round(prep.compression_pct, 1),
                          prep.cache_hit, prep.cache_tier)

    def _make(self, task, variant, model, condition, gen,
              orig, in_tok, out_tok, ms, saved, pct, hit, tier):
        passed, total = self.evaluator.evaluate(gen, task.test_cases, task.entry_point)
        cost = estimate_cost(model, in_tok, out_tok)
        return ExperimentRecord(
            task_id=task.task_id, variant=variant, model=model, condition=condition,
            original_tokens=orig, input_tokens=in_tok, output_tokens=out_tok,
            tokens_saved_compression=saved, compression_pct=pct,
            cache_hit=hit, cache_tier=tier, latency_ms=ms,
            estimated_cost_usd=cost, generated_code=gen,
            tests_passed=passed, tests_total=total,
            pass_at_1=passed/total if total > 0 else 0.0,
        )

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_summary(self):
        if not self.records:
            return

        print(f"\n{'='*76}")
        print("  CONDITION COMPARISON  (avg per request)")
        print(f"{'='*76}")

        for model in self.models:
            print(f"\n  Model: {model}")
            print(f"  {'Condition':<18} {'Tok Sent':>9} {'Cost/req':>10} {'Pass@1':>7} {'Cache%':>7}  {'vs Baseline'}")
            print(f"  {'─'*18} {'─'*9} {'─'*10} {'─'*7} {'─'*7}  {'─'*12}")
            baseline_cost = None
            for cond in CONDITIONS:
                recs = [r for r in self.records if r.model == model and r.condition == cond]
                if not recs: continue
                avg_tok  = sum(r.input_tokens      for r in recs) / len(recs)
                avg_cost = sum(r.estimated_cost_usd for r in recs) / len(recs)
                avg_pass = sum(r.pass_at_1         for r in recs) / len(recs)
                hit_pct  = sum(1 for r in recs if r.cache_hit) / len(recs) * 100
                if cond == "baseline":
                    baseline_cost = avg_cost; vs = "—"
                else:
                    pct = (baseline_cost - avg_cost) / baseline_cost * 100 if baseline_cost else 0
                    vs = f"{pct:+.1f}%"
                print(f"  {cond:<18} {avg_tok:>9.0f} ${avg_cost:>9.5f} {avg_pass:>7.1%} {hit_pct:>6.0f}%  {vs}")

        print(f"\n{'='*76}")
        print("  CACHE HITS BY VARIANT  (full_v2 only)")
        print(f"{'='*76}")
        print(f"  {'Variant':<16} {'Reqs':>5} {'Hits':>6} {'Hit%':>6}  {'Avg Tok Sent':>13}  {'Tok Saved vs Baseline':>21}")
        print(f"  {'─'*16} {'─'*5} {'─'*6} {'─'*6}  {'─'*13}  {'─'*21}")
        v2_recs = [r for r in self.records if r.condition == "full_v2"]
        for variant in ["original", "paraphrase_1", "paraphrase_2", "exact_repeat"]:
            vr = [r for r in v2_recs  if r.variant == variant]
            br = [r for r in self.records if r.condition == "baseline" and r.variant == variant]
            if not vr: continue
            hits     = sum(1 for r in vr if r.cache_hit)
            avg_tok  = sum(r.input_tokens for r in vr) / len(vr)
            avg_base = sum(r.input_tokens for r in br) / len(br) if br else 0
            saved    = avg_base - avg_tok
            print(f"  {variant:<16} {len(vr):>5} {hits:>6} {hits/len(vr)*100:>5.0f}%  {avg_tok:>13.0f}  {saved:>21.0f}")

        print(f"\n{'='*76}")
        print("  TOKEN SAVINGS BREAKDOWN  (vs baseline, avg per request)")
        print(f"{'='*76}")
        br = [r for r in self.records if r.condition == "baseline"]
        cr = [r for r in self.records if r.condition == "caveman"]
        vr = [r for r in self.records if r.condition == "full_v2"]
        if br:
            avg_orig      = sum(r.original_tokens  for r in br) / len(br)
            avg_base_sent = sum(r.input_tokens      for r in br) / len(br)
            print(f"  Original prompt size:       {avg_orig:.0f} tokens")
            print(f"  Baseline sends:             {avg_base_sent:.0f} tokens/req")
            if cr:
                avg_cav_sent  = sum(r.input_tokens for r in cr) / len(cr)
                cav_saved     = avg_base_sent - avg_cav_sent
                print(f"  caveman sends:              {avg_cav_sent:.0f} tokens/req  "
                      f"(─ {cav_saved:.0f} tokens, {cav_saved/avg_orig*100:.1f}% of original — BrevityMode only)")
            if vr:
                avg_v2_sent   = sum(r.input_tokens      for r in vr) / len(vr)
                avg_compr     = sum(r.tokens_saved_compression for r in vr) / len(vr)
                avg_cache     = max(0, avg_base_sent - avg_v2_sent - avg_compr)
                total_saved   = avg_base_sent - avg_v2_sent
                print(f"  full_v2 sends:              {avg_v2_sent:.0f} tokens/req")
                print(f"  ── Saved by BrevityMode:    see caveman row above")
                print(f"  ── Saved by compression:    {avg_compr:.0f} tokens  ({avg_compr/avg_orig*100:.1f}% of original)")
                print(f"  ── Saved by cache:          {avg_cache:.0f} tokens  ({avg_cache/avg_orig*100:.1f}% of original)")
                print(f"  Total saved (full_v2):      {total_saved:.0f} tokens  ({total_saved/avg_orig*100:.1f}% of original)")

        print(f"\n{'='*76}")
        print("  PROJECTED COST AT SCALE")
        print(f"{'='*76}")
        for model in self.models:
            br = [r for r in self.records if r.model == model and r.condition == "baseline"]
            cr = [r for r in self.records if r.model == model and r.condition == "caveman"]
            vr = [r for r in self.records if r.model == model and r.condition == "full_v2"]
            if not br: continue
            avg_b = sum(r.estimated_cost_usd for r in br) / len(br)
            avg_c = sum(r.estimated_cost_usd for r in cr) / len(cr) if cr else None
            avg_v = sum(r.estimated_cost_usd for r in vr) / len(vr) if vr else None
            red_c = (avg_b - avg_c) / avg_b * 100 if avg_c and avg_b else 0
            red_v = (avg_b - avg_v) / avg_b * 100 if avg_v and avg_b else 0
            print(f"\n  {model}")
            print(f"    caveman  {red_c:.1f}% cost reduction  |  full_v2  {red_v:.1f}% cost reduction")
            print(f"  {'Daily Volume':>16}  {'Baseline/day':>14}  {'caveman/day':>12}  {'full_v2/day':>12}  {'v2 Saved/mo':>13}")
            print(f"  {'─'*16}  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*13}")
            for scale in [1_000, 10_000, 100_000]:
                b_day = avg_b * scale
                c_day = avg_c * scale if avg_c else b_day
                v_day = avg_v * scale if avg_v else b_day
                saved_day = b_day - v_day
                print(f"  {scale:>16,}  ${b_day:>13.2f}  ${c_day:>11.2f}  ${v_day:>11.2f}  ${saved_day*30:>12.2f}")

    def save_results(self, path: str = "results/raw_resultsq.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
        print(f"\n[Saved] {len(self.records)} records → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",          action="store_true", default=True)
    parser.add_argument("--real",             action="store_true")
    parser.add_argument("--openai-key",       default="")
    parser.add_argument("--anthropic-key",    default="")
    parser.add_argument("--n-base-tasks",     type=int, default=10)
    parser.add_argument("--models",           nargs="+", default=["mock-model"])
    parser.add_argument(
        "--brevity-position",
        default="pre_precall",
        choices=list(BREVITY_POSITIONS),
        help=(
            "Where in the pipeline BrevityMode fires. "
            "'pre_precall' (default, matches shahira): replace the verbose system prompt "
            "BEFORE PreCallLayer so compression stages work on a smaller prompt and "
            "savings compound. "
            "'post_precall': PreCallLayer runs on the original verbose prompt first, "
            "THEN the system portion is replaced at assembly time. "
            "Use 'post_precall' to isolate BrevityMode savings independently of "
            "PreCallLayer interactions."
        ),
    )
    args = parser.parse_args()

    loader = DatasetLoader()
    base_tasks = loader.load_all(n_each=args.n_base_tasks)

    runner = ExperimentRunner(
        models=args.models,
        dry_run=not args.real,
        openai_key=os.getenv("OPENAI_API_KEY", args.openai_key or ""),
        anthropic_key=os.getenv("ANTHROPIC_API_KEY", args.anthropic_key or ""),
        brevity_position=args.brevity_position,
    )
    runner.run(base_tasks)
    runner.print_summary()
    runner.save_results()