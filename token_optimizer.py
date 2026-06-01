"""
TokenOptimizer — Core library for reducing LLM token usage in code generation.

Techniques:
  1. Rule-based compression  (whitespace, redundancy, markdown cleanup)
  2. Semantic deduplication  (embedding-based near-duplicate removal)
  3. Dynamic few-shot pruning (TF-IDF relevance ranking of examples)
  4. Instruction distillation (condense verbose system prompts)
"""

import re
import math
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# 1. RULE-BASED COMPRESSOR
# ──────────────────────────────────────────────────────────────────────────────

class RuleBasedCompressor:
    """Fast, deterministic compression. No external dependencies."""

    def compress(self, text: str) -> str:
        text = self._remove_extra_whitespace(text)
        text = self._strip_markdown_artifacts(text)
        text = self._collapse_repeated_punctuation(text)
        text = self._remove_html_tags(text)
        text = self._truncate_long_comments(text)
        return text.strip()

    def _remove_extra_whitespace(self, text: str) -> str:
        # Collapse 3+ blank lines to 1
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip trailing spaces on each line
        text = '\n'.join(line.rstrip() for line in text.splitlines())
        return text

    def _strip_markdown_artifacts(self, text: str) -> str:
        # Remove fenced code block delimiters if they wrap the entire text
        text = re.sub(r'^```\w*\n', '', text)
        text = re.sub(r'\n```$', '', text)
        # Remove bold/italic markers that don't add meaning in prompts
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        return text

    def _collapse_repeated_punctuation(self, text: str) -> str:
        # Collapse "....." to "..."
        text = re.sub(r'\.{4,}', '...', text)
        # Collapse repeated dashes used as separators
        text = re.sub(r'-{4,}', '---', text)
        text = re.sub(r'={4,}', '===', text)
        return text

    def _remove_html_tags(self, text: str) -> str:
        return re.sub(r'<[^>]+>', '', text)

    def _truncate_long_comments(self, text: str, max_comment_lines: int = 5) -> str:
        """
        Truncate docstrings / block comments that exceed max_comment_lines.
        Keeps the first max_comment_lines and appends a truncation note.
        """
        lines = text.splitlines()
        result = []
        in_docstring = False
        docstring_lines = []
        docstring_char = None

        for line in lines:
            stripped = line.strip()
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    char = stripped[:3]
                    # Check if it closes on the same line
                    rest = stripped[3:]
                    if char in rest:
                        result.append(line)
                    else:
                        in_docstring = True
                        docstring_char = char
                        docstring_lines = [line]
                else:
                    result.append(line)
            else:
                docstring_lines.append(line)
                if docstring_char in stripped:
                    in_docstring = False
                    if len(docstring_lines) > max_comment_lines + 2:
                        kept = docstring_lines[:max_comment_lines + 1]
                        close = docstring_lines[-1]
                        indent = len(line) - len(line.lstrip())
                        kept.append(' ' * indent + '[...truncated...]')
                        kept.append(close)
                        result.extend(kept)
                    else:
                        result.extend(docstring_lines)
                    docstring_lines = []

        if in_docstring:
            result.extend(docstring_lines)

        return '\n'.join(result)


# ──────────────────────────────────────────────────────────────────────────────
# 2. SEMANTIC DEDUPLICATOR  (optional — requires numpy)
# ──────────────────────────────────────────────────────────────────────────────

class SemanticDeduplicator:
    """
    Splits context into paragraphs and removes near-duplicates using
    TF-IDF cosine similarity (no sentence-transformers needed).
    Falls back gracefully if numpy is unavailable.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold

    def deduplicate(self, text: str) -> str:
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        if len(paragraphs) <= 1:
            return text

        try:
            kept = self._filter_similar(paragraphs)
        except Exception:
            kept = paragraphs  # fallback: no dedup

        return '\n\n'.join(kept)

    def _tfidf_vectors(self, docs: List[str]):
        """Minimal TF-IDF without sklearn."""
        import math

        def tokenize(d):
            return re.findall(r'\w+', d.lower())

        tokenized = [tokenize(d) for d in docs]
        vocab = sorted(set(w for t in tokenized for w in t))
        vocab_idx = {w: i for i, w in enumerate(vocab)}
        N = len(docs)

        # Document frequency
        df = [0] * len(vocab)
        for t in tokenized:
            for w in set(t):
                if w in vocab_idx:
                    df[vocab_idx[w]] += 1

        # TF-IDF matrix as list of dicts (sparse)
        vectors = []
        for t in tokenized:
            vec = {}
            tf_counts = {}
            for w in t:
                tf_counts[w] = tf_counts.get(w, 0) + 1
            for w, cnt in tf_counts.items():
                if w in vocab_idx:
                    tf = cnt / len(t)
                    idf = math.log((N + 1) / (df[vocab_idx[w]] + 1)) + 1
                    vec[vocab_idx[w]] = tf * idf
            vectors.append(vec)

        return vectors

    def _cosine(self, a: dict, b: dict) -> float:
        dot = sum(a.get(k, 0) * v for k, v in b.items())
        norm_a = math.sqrt(sum(v**2 for v in a.values()))
        norm_b = math.sqrt(sum(v**2 for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _filter_similar(self, paragraphs: List[str]) -> List[str]:
        vectors = self._tfidf_vectors(paragraphs)
        kept_indices = []
        for i, vec_i in enumerate(vectors):
            is_duplicate = False
            for j in kept_indices:
                if self._cosine(vec_i, vectors[j]) >= self.threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_indices.append(i)
        return [paragraphs[i] for i in kept_indices]


# ──────────────────────────────────────────────────────────────────────────────
# 3. FEW-SHOT PRUNER
# ──────────────────────────────────────────────────────────────────────────────

class FewShotPruner:
    """
    Given a list of (input, output) examples and a query,
    rank by TF-IDF similarity and keep only top-k.
    """

    def __init__(self, top_k: int = 2):
        self.top_k = top_k
        self._dedup = SemanticDeduplicator(similarity_threshold=0.0)  # reuse vectorizer

    def prune(self, examples: List[Tuple[str, str]], query: str) -> List[Tuple[str, str]]:
        if len(examples) <= self.top_k:
            return examples

        # Combine input+output for each example into a single doc
        docs = [f"{inp} {out}" for inp, out in examples]
        docs.append(query)

        vectors = self._dedup._tfidf_vectors(docs)
        query_vec = vectors[-1]
        example_vecs = vectors[:-1]

        scored = [
            (self._dedup._cosine(query_vec, ev), i)
            for i, ev in enumerate(example_vecs)
        ]
        scored.sort(reverse=True)
        top_indices = [i for _, i in scored[:self.top_k]]
        # Preserve original order
        top_indices.sort()
        return [examples[i] for i in top_indices]


# ──────────────────────────────────────────────────────────────────────────────
# 4. INSTRUCTION DISTILLER
# ──────────────────────────────────────────────────────────────────────────────

class InstructionDistiller:
    """
    Rule-based compression of system prompts:
    - Remove filler phrases
    - Collapse repeated instructions
    - Shorten verbose preambles
    """

    FILLER_PATTERNS = [
        r'please\s+', r'kindly\s+', r'make sure to\s+', r'you should\s+',
        r'it is important that\s+', r'note that\s+', r'keep in mind that\s+',
        r'as an ai language model,?\s*', r'as a helpful assistant,?\s*',
        r'i\'d like you to\s+', r'could you please\s+',
    ]

    def distill(self, system_prompt: str) -> str:
        prompt = system_prompt
        for pattern in self.FILLER_PATTERNS:
            prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)
        # Collapse duplicate sentences
        sentences = re.split(r'(?<=[.!?])\s+', prompt)
        seen = set()
        unique = []
        for s in sentences:
            key = re.sub(r'\s+', ' ', s.lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return ' '.join(unique).strip()


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN TOKEN OPTIMIZER  (orchestrates all techniques)
# ──────────────────────────────────────────────────────────────────────────────

class TokenOptimizer:
    """
    Main optimizer — composes all techniques into a single pipeline.

    Usage:
        optimizer = TokenOptimizer()
        result = optimizer.optimize(
            system_prompt="You are a helpful coding assistant...",
            user_prompt="Write a Python function that...",
            context="Here is some background: ...",
            examples=[("input1", "output1"), ("input2", "output2")],
            query="Write a function to sort a list"
        )
        print(result.optimized_prompt)
        print(result.stats)
    """

    def __init__(
        self,
        use_rule_based: bool = True,
        use_semantic_dedup: bool = True,
        use_few_shot_pruning: bool = True,
        use_instruction_distill: bool = True,
        few_shot_top_k: int = 2,
        dedup_threshold: float = 0.85,
    ):
        self.compressor = RuleBasedCompressor() if use_rule_based else None
        self.deduplicator = SemanticDeduplicator(dedup_threshold) if use_semantic_dedup else None
        self.pruner = FewShotPruner(few_shot_top_k) if use_few_shot_pruning else None
        self.distiller = InstructionDistiller() if use_instruction_distill else None

    def optimize(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        context: str = "",
        examples: Optional[List[Tuple[str, str]]] = None,
        query: str = "",
    ) -> "OptimizationResult":

        original_tokens = self._count_tokens(system_prompt + user_prompt + context)

        # Step 1: Distill system prompt
        opt_system = self.distiller.distill(system_prompt) if self.distiller and system_prompt else system_prompt

        # Step 2: Rule-based compression on user prompt and context
        opt_user = self.compressor.compress(user_prompt) if self.compressor and user_prompt else user_prompt
        opt_context = self.compressor.compress(context) if self.compressor and context else context

        # Step 3: Semantic deduplication on context
        if self.deduplicator and opt_context:
            opt_context = self.deduplicator.deduplicate(opt_context)

        # Step 4: Prune few-shot examples
        opt_examples = examples or []
        if self.pruner and opt_examples and query:
            opt_examples = self.pruner.prune(opt_examples, query)

        # Assemble final prompt
        parts = []
        if opt_system:
            parts.append(f"[SYSTEM] {opt_system}")
        if opt_context:
            parts.append(f"[CONTEXT]\n{opt_context}")
        if opt_examples:
            ex_str = "\n".join(f"Input: {i}\nOutput: {o}" for i, o in opt_examples)
            parts.append(f"[EXAMPLES]\n{ex_str}")
        if opt_user:
            parts.append(f"[TASK] {opt_user}")

        final_prompt = "\n\n".join(parts)
        optimized_tokens = self._count_tokens(final_prompt)

        return OptimizationResult(
            original_prompt="\n".join([system_prompt, context, user_prompt]),
            optimized_prompt=final_prompt,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            examples_kept=len(opt_examples),
            examples_original=len(examples) if examples else 0,
        )

    def _count_tokens(self, text: str) -> int:
        """Approximate token count: ~4 chars per token (OpenAI rule of thumb)."""
        return max(1, len(text) // 4)


# ──────────────────────────────────────────────────────────────────────────────
# 6. RESULT OBJECT
# ──────────────────────────────────────────────────────────────────────────────

class OptimizationResult:
    def __init__(
        self,
        original_prompt: str,
        optimized_prompt: str,
        original_tokens: int,
        optimized_tokens: int,
        examples_kept: int,
        examples_original: int,
    ):
        self.original_prompt = original_prompt
        self.optimized_prompt = optimized_prompt
        self.original_tokens = original_tokens
        self.optimized_tokens = optimized_tokens
        self.examples_kept = examples_kept
        self.examples_original = examples_original

    @property
    def token_reduction(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.original_tokens - self.optimized_tokens) / self.original_tokens

    @property
    def stats(self) -> dict:
        return {
            "original_tokens": self.original_tokens,
            "optimized_tokens": self.optimized_tokens,
            "tokens_saved": self.original_tokens - self.optimized_tokens,
            "reduction_pct": round(self.token_reduction * 100, 2),
            "examples_original": self.examples_original,
            "examples_kept": self.examples_kept,
        }

    def __repr__(self):
        s = self.stats
        return (
            f"OptimizationResult("
            f"tokens {s['original_tokens']}→{s['optimized_tokens']} "
            f"({s['reduction_pct']}% saved), "
            f"examples {s['examples_original']}→{s['examples_kept']})"
        )
