"""
token_optimizer_v2.py
─────────────────────
Token Optimizer Library — v2

A zero-dependency library for reducing LLM token usage and cost.
Import the modules you need; use TokenOptimizerV2 for the full pipeline.

Library structure
─────────────────
  token_optimizer_v2.py          ← YOU ARE HERE (library entry point)
  ├── token_optimizer.py         ← v1 PreCallLayer components (unchanged)
  └── cache_layer.py             ← CacheLayer components

Exported API
────────────
  # Full pipeline
  from token_optimizer_v2 import TokenOptimizerV2

  # Individual layers (use standalone if needed)
  from token_optimizer_v2 import (
      # PreCallLayer
      RuleBasedCompressor,
      SemanticDeduplicator,
      FewShotPruner,
      InstructionDistiller,

      # CacheLayer
      PrefixCacheInjector,
      SemanticResponseCache,
      CacheLayer,

      # OutputLayer (BrevityMode)
      BREVITY_CODING_INSTRUCTIONS,
      BREVITY_POSITIONS,
      build_system_prompt,

      # v1 backward compat
      TokenOptimizer,

      # Result types
      PrepareResult,
      V2CallResult,
      OptimizationResult,
  )

Pipeline (per call)
───────────────────
  Input: system_prompt, context, examples, task
      │
      ▼  ◄── brevity_position="pre_precall" (default): _build_system() fires HERE
  ┌─────────────────────────────┐
  │  OutputLayer  (BrevityMode) │  REPLACES verbose system prompt — saves ~50
  │  8. build_system_prompt()   │  tokens before any other technique runs
  └────────────┬────────────────┘
               │ tight system prompt
               ▼
  ┌─────────────────────────────┐
  │  PreCallLayer               │
  │  1. InstructionDistiller    │  strips filler phrases from system prompt
  │  2. RuleBasedCompressor     │  whitespace / markdown cleanup
  │  3. SemanticDeduplicator    │  removes near-duplicate context paragraphs
  │  4. FewShotPruner           │  keeps only top-k relevant examples
  └────────────┬────────────────┘
               │ compressed prompt
               ▼  ◄── brevity_position="post_precall": _build_system() fires HERE
  ┌─────────────────────────────┐          (replaces system portion at assembly)
  │  CacheLayer                 │
  │  5. SemanticResponseCache   │  exact + semantic lookup → skip LLM on hit
  │  6. PrefixCacheInjector     │  reorders prompt for provider KV-cache
  └────────────┬────────────────┘
               │ final prompt  (or cached response)
               ▼
           LLM API call
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT LAYER — BrevityMode (technique #8)
# ══════════════════════════════════════════════════════════════════════════════
#
# BrevityMode controls what the LLM is *instructed to generate* — it is always
# an input-side instruction, never a post-processor of LLM output.  The name
# "output layer" in the shahira docstring refers to the layer that governs LLM
# *output format*, not something that runs after the LLM responds.
#
# brevity_position controls WHERE in the input pipeline the instruction lands:
#
#   "pre_precall"  (default, matches shahira)
#     _build_system() runs BEFORE PreCallLayer.
#     The tight system prompt is what InstructionDistiller, RuleBasedCompressor
#     etc. all operate on — so later stages see a smaller, already-clean prompt.
#     Saves ~50 tokens; those savings compound with every subsequent technique.
#
#   "post_precall"
#     PreCallLayer runs on the original verbose system prompt first, then the
#     result is REPLACED with build_system_prompt() at assembly time.
#     PreCallLayer therefore processes more tokens than necessary, but the
#     final assembled prompt is identical.  Use this position to measure
#     the pure token-count difference without the compounding effect.
#
# Behaviour matrix for _build_system() (same in both positions):
#   distill=False, brevity="off"  → passthrough (true baseline, fair comparison)
#   distill=False, brevity=X      → replace with build_system_prompt(X)
#   distill=True,  brevity="off"  → distill base, keep its output instruction
#   distill=True,  brevity=X      → replace with build_system_prompt(X)

BREVITY_POSITIONS = ("pre_precall", "post_precall")

BREVITY_CODING_INSTRUCTIONS: Dict[str, str] = {
    # "off"   — faithful passthrough for baseline; no replacement
    "off":   "Return only the complete function code — no explanations, no markdown, no preamble.",
    # "lite"  — one short sentence; suits models that follow terse instructions
    "lite":  "Return only the complete function code. No markdown. No explanation.",
    # "full"  — explicit about the def line; prevents body-only responses on weak models
    "full":  "Return the complete function including the def line. No explanation. No markdown.",
    # "ultra" — fewest tokens; still keeps "def line" guard for GPT-4o-mini / Haiku
    "ultra": "Return complete function code. def line required. No markdown.",
}

# Stable role prefix shared by all non-baseline brevity modes.
# Replaces the ~10-line verbose system prompt (saves ~50 tokens before any
# other optimisation runs).
_CODING_ROLE = "Python engineer. Write clean, correct, idiomatic Python. Handle edge cases."


def build_system_prompt(brevity_mode: str = "off") -> str:
    """
    Return a tight, brevity-fused system prompt.
    The output instruction is FUSED (not appended) so it never double-counts.
    """
    instr = BREVITY_CODING_INSTRUCTIONS.get(brevity_mode, BREVITY_CODING_INSTRUCTIONS["off"])
    return f"{_CODING_ROLE}\n{instr}"


# ── Re-export all v1 PreCallLayer components ──────────────────────────────────
from token_optimizer import (
    RuleBasedCompressor,
    SemanticDeduplicator,
    FewShotPruner,
    InstructionDistiller,
    TokenOptimizer,        # v1 — re-exported for backward compatibility
    OptimizationResult,    # v1 result type
)

# ── Re-export all CacheLayer components ───────────────────────────────────────
from cache_layer import (
    CacheLayer,
    CacheCallResult,
    CacheStats,
    CachedEntry,
    PrefixCacheInjector,
    SemanticResponseCache,
)


# ──────────────────────────────────────────────────────────────────────────────
# RESULT TYPES
# ──────────────────────────────────────────────────────────────────────────────

class PrepareResult:
    """
    Returned by TokenOptimizerV2.prepare().

    Workflow:
        result = optimizer.prepare(system_prompt=..., context=...,
                                   examples=..., task=...)

        if result.cache_hit:
            use result.response directly   # free — no LLM call
        else:
            raw = my_llm(result.prompt)    # send compressed prompt to LLM
            optimizer.record(result, raw_response=raw, tokens_used=N)
    """

    def __init__(
        self,
        prompt: str,
        cache_hit: bool,
        response: Optional[str],
        cache_tier: Optional[str],       # "exact" | "semantic" | None
        prefix_cache_eligible: bool,
        pre_call_stats: dict,
        original_tokens: int,
        compressed_tokens: int,
        _cache_key: str = "",
    ):
        self.prompt = prompt
        self.cache_hit = cache_hit
        self.response = response
        self.cache_tier = cache_tier
        self.prefix_cache_eligible = prefix_cache_eligible
        self.pre_call_stats = pre_call_stats
        self.original_tokens = original_tokens
        self.compressed_tokens = compressed_tokens
        self._cache_key = _cache_key

    @property
    def compression_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.original_tokens - self.compressed_tokens) / self.original_tokens * 100

    @property
    def tokens_saved_by_compression(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    def __repr__(self) -> str:
        if self.cache_hit:
            return f"PrepareResult(cache_hit=True, tier={self.cache_tier})"
        return (
            f"PrepareResult(cache_hit=False, "
            f"tokens={self.original_tokens}→{self.compressed_tokens} "
            f"({self.compression_pct:.1f}% saved), "
            f"prefix_cached={self.prefix_cache_eligible})"
        )


class V2CallResult:
    """
    Returned by TokenOptimizerV2.call().
    Contains the LLM response plus full stats from both layers.
    """

    def __init__(
        self,
        response: str,
        from_cache: bool,
        cache_tier: Optional[str],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        original_tokens: int,
        compressed_tokens: int,
        prefix_cache_eligible: bool,
        pre_call_stats: dict,
    ):
        self.response = response
        self.from_cache = from_cache
        self.cache_tier = cache_tier
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.original_tokens = original_tokens
        self.compressed_tokens = compressed_tokens
        self.prefix_cache_eligible = prefix_cache_eligible
        self.pre_call_stats = pre_call_stats

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def compression_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.original_tokens - self.compressed_tokens) / self.original_tokens * 100

    @property
    def tokens_saved_by_compression(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def stats(self) -> dict:
        return {
            "from_cache":            self.from_cache,
            "cache_tier":            self.cache_tier,
            "original_tokens":       self.original_tokens,
            "compressed_tokens":     self.compressed_tokens,
            "compression_pct":       round(self.compression_pct, 1),
            "input_tokens_sent":     self.input_tokens,
            "output_tokens":         self.output_tokens,
            "total_tokens":          self.total_tokens,
            "latency_ms":            round(self.latency_ms, 1),
            "prefix_cache_eligible": self.prefix_cache_eligible,
            "pre_call":              self.pre_call_stats,
        }

    def __repr__(self) -> str:
        src = f"cache({self.cache_tier})" if self.from_cache else "LLM"
        return (
            f"V2CallResult(source={src}, "
            f"tokens {self.original_tokens}→{self.compressed_tokens}→{self.input_tokens} sent, "
            f"latency={self.latency_ms:.0f}ms)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TOKEN OPTIMIZER V2  —  main library class
# ──────────────────────────────────────────────────────────────────────────────

class TokenOptimizerV2:
    """
    Full token-optimization pipeline: PreCallLayer → CacheLayer → LLM.

    Parameters
    ──────────
    OutputLayer (BrevityMode):
      brevity_mode            str    Controls system-prompt replacement strategy.
                                     "off"   → passthrough (true baseline)
                                     "lite"  → short terse instruction
                                     "full"  → explicit def-line guard (default)
                                     "ultra" → fewest tokens, still safe
                                     REPLACES the verbose system prompt instead of
                                     appending — avoids the ~51-token append penalty.
      brevity_position        str    Where in the pipeline BrevityMode fires.
                                     "pre_precall"  (default) — replace system prompt
                                       BEFORE PreCallLayer; PreCallLayer then works on
                                       a smaller prompt and savings compound.
                                     "post_precall" — PreCallLayer runs on the original
                                       verbose prompt first; system replaced at assembly.
                                       Use to isolate BrevityMode savings independently.

    PreCallLayer:
      use_rule_based          bool   Whitespace/markdown compression.    Default: True
      use_semantic_dedup      bool   Remove near-duplicate paragraphs.   Default: False
      use_few_shot_pruning    bool   Keep only top-k relevant examples.  Default: True
      use_instruction_distill bool   Strip filler from system prompt.    Default: True
      few_shot_top_k          int    Number of examples to keep.         Default: 2
      dedup_threshold         float  Cosine cutoff for paragraph dedup.  Default: 0.85

    CacheLayer:
      use_response_cache      bool   Client-side semantic cache.         Default: True
      use_prefix_injection    bool   Reorder for provider KV-cache.      Default: True
      cache_size              int    Max entries in response cache.       Default: 1000
      similarity_threshold    float  Cosine cutoff for cache hits.       Default: 0.92
      cache_ttl_seconds       float  Per-entry TTL; None = no expiry.    Default: None
      min_prefix_tokens       int    Min tokens for prefix caching.      Default: 1024
      cost_per_1k_tokens      float  For cost-saving estimates.          Default: 0.003

    Two usage patterns
    ──────────────────
    Pattern A — all-in-one (simplest):
        result = optimizer.call(llm_fn=my_llm, system_prompt=..., task=...)
        print(result.response, result.stats)

    Pattern B — two-step (full control over the LLM call):
        prep = optimizer.prepare(system_prompt=..., task=...)
        if prep.cache_hit:
            response = prep.response
        else:
            response, in_tok, out_tok, ms = my_llm(prep.prompt)
            optimizer.record(prep, response, tokens_used=in_tok)
    """

    def __init__(
        self,
        # OutputLayer — BrevityMode
        brevity_mode: str = "ultra",           # "off"|"lite"|"full"|"ultra"
        brevity_position: str = "pre_precall",  # "pre_precall"|"post_precall"
        # PreCallLayer
        use_rule_based: bool = True,
        use_semantic_dedup: bool = True,
        use_few_shot_pruning: bool = True,
        use_instruction_distill: bool = True,
        few_shot_top_k: int = 2,
        dedup_threshold: float = 0.85,
        # CacheLayer
        use_response_cache: bool = True,
        use_prefix_injection: bool = True,
        cache_size: int = 1_000,
        similarity_threshold: float = 0.92,
        cache_ttl_seconds: Optional[float] = None,
        min_prefix_tokens: int = 1_024,
        cost_per_1k_tokens: float = 0.003,
    ):
        assert brevity_mode in BREVITY_CODING_INSTRUCTIONS, (
            f"brevity_mode must be one of {list(BREVITY_CODING_INSTRUCTIONS)}; got {brevity_mode!r}"
        )
        assert brevity_position in BREVITY_POSITIONS, (
            f"brevity_position must be one of {list(BREVITY_POSITIONS)}; got {brevity_position!r}"
        )
        # OutputLayer
        self.brevity_mode     = brevity_mode
        self.brevity_position = brevity_position

        # PreCallLayer
        self.compressor   = RuleBasedCompressor()                 if use_rule_based           else None
        self.deduplicator = SemanticDeduplicator(dedup_threshold) if use_semantic_dedup        else None
        self.pruner       = FewShotPruner(few_shot_top_k)         if use_few_shot_pruning      else None
        self.distiller    = InstructionDistiller()                if use_instruction_distill   else None

        # CacheLayer
        self.cache_layer = CacheLayer(
            semantic_cache_size=cache_size,
            similarity_threshold=similarity_threshold,
            min_prefix_tokens=min_prefix_tokens,
            ttl_seconds=cache_ttl_seconds,
            cost_per_1k_tokens=cost_per_1k_tokens,
            use_semantic_cache=use_response_cache,
            use_prefix_injection=use_prefix_injection,
        )

        # Session counters
        self._session_original_tokens: int = 0
        self._session_compressed_tokens: int = 0
        self._session_calls: int = 0
        self._session_cache_hits: int = 0

    # ── Pattern A: all-in-one ─────────────────────────────────────────────────

    def call(
        self,
        llm_fn: Callable[[str], tuple],
        system_prompt: str = "",
        context: str = "",
        examples: Optional[List[Tuple[str, str]]] = None,
        task: str = "",
    ) -> V2CallResult:
        """
        Run the full pipeline and return a V2CallResult.

        llm_fn must accept a prompt string and return one of:
          (response_str, input_tokens, output_tokens, latency_ms)
          (response_str, input_tokens)
          response_str
        """
        prep = self.prepare(system_prompt=system_prompt, context=context,
                            examples=examples or [], task=task)

        if prep.cache_hit:
            return V2CallResult(
                response=prep.response, from_cache=True,
                cache_tier=prep.cache_tier, input_tokens=0, output_tokens=0,
                latency_ms=0.0, original_tokens=prep.original_tokens,
                compressed_tokens=prep.compressed_tokens,
                prefix_cache_eligible=False, pre_call_stats=prep.pre_call_stats,
            )

        raw = llm_fn(prep.prompt)
        if isinstance(raw, tuple) and len(raw) == 4:
            response, input_tokens, output_tokens, latency_ms = raw
        elif isinstance(raw, tuple) and len(raw) == 2:
            response, input_tokens = raw
            output_tokens, latency_ms = 0, 0.0
        else:
            response, input_tokens, output_tokens, latency_ms = str(raw), 0, 0, 0.0

        self.record(prep, response, tokens_used=input_tokens)

        return V2CallResult(
            response=response, from_cache=False, cache_tier=None,
            input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=latency_ms, original_tokens=prep.original_tokens,
            compressed_tokens=prep.compressed_tokens,
            prefix_cache_eligible=prep.prefix_cache_eligible,
            pre_call_stats=prep.pre_call_stats,
        )

    # ── Pattern B: two-step ───────────────────────────────────────────────────

    def prepare(
        self,
        system_prompt: str = "",
        context: str = "",
        examples: Optional[List[Tuple[str, str]]] = None,
        task: str = "",
    ) -> PrepareResult:
        """
        Run PreCallLayer + cache lookup without calling any LLM.
        Returns a PrepareResult with either a ready response (cache hit)
        or a compressed prompt to send to your LLM.
        """
        examples = examples or []
        original_tokens = self._count_tokens(
            system_prompt + context + task +
            "".join(i + o for i, o in examples)
        )

        # ── OutputLayer position: "pre_precall" ───────────────────────────────
        # Replace verbose system prompt BEFORE PreCallLayer so that distiller,
        # compressor etc. all operate on the already-tight prompt.  Savings
        # compound: a smaller input means fewer filler phrases to strip, fewer
        # whitespace passes, etc.  This is the default and matches shahira.
        if self.brevity_position == "pre_precall":
            sys_for_pipeline = self._build_system(system_prompt)
        else:
            sys_for_pipeline = system_prompt  # PreCallLayer gets the raw verbose prompt

        # PreCallLayer
        opt_system   = self._distill(sys_for_pipeline)
        opt_context  = self._compress_and_dedup(context)
        opt_examples = self._prune_examples(examples, task)
        opt_task     = self._compress(task)

        # ── OutputLayer position: "post_precall" ──────────────────────────────
        # PreCallLayer ran on the original verbose prompt; NOW replace the system
        # portion with the tight brevity-fused version at assembly time.
        # Final assembled prompt is equivalent but PreCallLayer did extra work.
        if self.brevity_position == "post_precall":
            opt_system = self._build_system(system_prompt)

        compressed_tokens = self._count_tokens(
            opt_system + opt_context + opt_task +
            "".join(i + o for i, o in opt_examples)
        )
        pre_call_stats = {
            "original_tokens":   original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_pct":   round((original_tokens - compressed_tokens) / max(original_tokens, 1) * 100, 1),
            "examples_original": len(examples),
            "examples_kept":     len(opt_examples),
            "brevity_mode":      self.brevity_mode,
            "brevity_position":  self.brevity_position,
        }

        self._session_original_tokens   += original_tokens
        self._session_compressed_tokens += compressed_tokens
        self._session_calls             += 1

        # CacheLayer: lookup
        cache_key = CacheLayer._build_cache_key(opt_system, opt_context, opt_examples, opt_task)
        cached_response: Optional[str] = None
        cache_tier: Optional[str] = None

        if self.cache_layer.response_cache:
            cached_response = self.cache_layer.response_cache.get(cache_key)
            if cached_response is not None:
                h = self.cache_layer.response_cache._hash(cache_key)
                cache_tier = "exact" if h in self.cache_layer.response_cache._store else "semantic"
                self._session_cache_hits += 1

        if cached_response is not None:
            return PrepareResult(
                prompt="", cache_hit=True, response=cached_response,
                cache_tier=cache_tier, prefix_cache_eligible=False,
                pre_call_stats=pre_call_stats, original_tokens=original_tokens,
                compressed_tokens=compressed_tokens, _cache_key=cache_key,
            )

        # CacheLayer: build prompt
        if self.cache_layer.prefix_injector:
            prompt_str, cache_eligible = self.cache_layer.prefix_injector.build_prompt_string(
                system_prompt=opt_system, shared_context=opt_context,
                examples=opt_examples, task=opt_task,
            )
        else:
            prompt_str = self._assemble(opt_system, opt_context, opt_examples, opt_task)
            cache_eligible = False

        return PrepareResult(
            prompt=prompt_str, cache_hit=False, response=None,
            cache_tier=None, prefix_cache_eligible=cache_eligible,
            pre_call_stats=pre_call_stats, original_tokens=original_tokens,
            compressed_tokens=compressed_tokens, _cache_key=cache_key,
        )

    def record(
        self,
        prepare_result: PrepareResult,
        raw_response: str,
        tokens_used: int = 0,
    ) -> None:
        """Store an LLM response in the cache after a successful call."""
        if self.cache_layer.response_cache and not prepare_result.cache_hit:
            self.cache_layer.response_cache.put(
                prepare_result._cache_key,
                raw_response,
                token_count=tokens_used or prepare_result.compressed_tokens,
            )

    # ── Stats & lifecycle ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Aggregated stats across all calls in this session."""
        total_orig = max(self._session_original_tokens, 1)
        saved_compression = self._session_original_tokens - self._session_compressed_tokens

        cache_stats: dict = {}
        if self.cache_layer.response_cache:
            cs = self.cache_layer.response_cache.stats
            cache_stats = {
                "size":                  len(self.cache_layer.response_cache),
                "hit_rate":              round(cs.hit_rate, 3),
                "hits":                  cs.hits,
                "misses":                cs.misses,
                "evictions":             cs.evictions,
                "tokens_saved_by_cache": cs.total_tokens_saved,
                "cost_saved_usd":        round(cs.total_cost_saved_usd, 5),
            }

        return {
            "session": {
                "total_calls":              self._session_calls,
                "cache_hits":               self._session_cache_hits,
                "original_tokens":          self._session_original_tokens,
                "compressed_tokens":        self._session_compressed_tokens,
                "tokens_saved_compression": saved_compression,
                "compression_pct":          round(saved_compression / total_orig * 100, 1),
            },
            "cache": cache_stats,
        }

    def reset_stats(self) -> None:
        """Reset session counters (does NOT clear the cache)."""
        self._session_original_tokens = 0
        self._session_compressed_tokens = 0
        self._session_calls = 0
        self._session_cache_hits = 0

    def clear_cache(self) -> None:
        """Wipe the response cache entirely."""
        if self.cache_layer.response_cache:
            self.cache_layer.response_cache.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_system(self, base: str) -> str:
        """
        OutputLayer: replace or pass through the system prompt.
        Called either before PreCallLayer (pre_precall) or after it
        (post_precall) depending on self.brevity_position — see prepare().

        Behaviour matrix:
          distill=False, brevity="off"  → passthrough base unchanged (true baseline)
          distill=False, brevity=X      → replace with build_system_prompt(X)
          distill=True,  brevity="off"  → distill base, keep its output instruction
          distill=True,  brevity=X      → replace with build_system_prompt(X) (tightest)
        """
        if self.distiller is None and self.brevity_mode == "off":
            return base or ""   # true passthrough
        return build_system_prompt(self.brevity_mode)

    def _distill(self, text: str) -> str:
        return self.distiller.distill(text) if self.distiller and text else text

    def _compress(self, text: str) -> str:
        return self.compressor.compress(text) if self.compressor and text else text

    def _compress_and_dedup(self, text: str) -> str:
        text = self._compress(text)
        return self.deduplicator.deduplicate(text) if self.deduplicator and text else text

    def _prune_examples(self, examples: List[Tuple[str, str]], query: str) -> List[Tuple[str, str]]:
        return self.pruner.prune(examples, query) if self.pruner and examples and query else examples

    @staticmethod
    def _assemble(system: str, context: str, examples: List[Tuple[str, str]], task: str) -> str:
        parts = []
        if system:   parts.append(f"[SYSTEM]\n{system}")
        if context:  parts.append(f"[CONTEXT]\n{context}")
        if examples: parts.append("[EXAMPLES]\n" + "\n".join(f"Input: {i}\nOutput: {o}" for i, o in examples))
        if task:     parts.append(f"[TASK]\n{task}")
        return "\n\n".join(parts)

    @staticmethod
    def _count_tokens(text: str) -> int:
        return max(1, len(text) // 4)