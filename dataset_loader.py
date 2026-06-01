"""
dataset_loader.py — Loads HumanEval and MBPP datasets for experiments.

We include a BUILT-IN subset so you can run everything without downloading.
For the full datasets, we also show how to load from HuggingFace.
"""

import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CodeTask:
    task_id: str
    source: str                   # "humaneval" | "mbpp" | "custom"
    prompt: str                   # The instruction / problem statement
    canonical_solution: str       # Reference solution
    test_cases: List[str]         # List of assert statements
    entry_point: str              # Function name to test


# ──────────────────────────────────────────────────────────────────────────────
# BUILT-IN MINI DATASET (20 tasks, no internet needed)
# ──────────────────────────────────────────────────────────────────────────────

BUILTIN_TASKS = [
    CodeTask(
        task_id="HE-1",
        source="humaneval",
        prompt="""Write a Python function called `has_close_elements` that takes a list of numbers
and a threshold value. Return True if any two numbers in the list are closer to
each other than the given threshold, otherwise return False.

Example:
    has_close_elements([1.0, 2.0, 3.0], 0.5) -> False
    has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) -> True""",
        canonical_solution="""def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False""",
        test_cases=[
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True",
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False",
            "assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True",
        ],
        entry_point="has_close_elements",
    ),
    CodeTask(
        task_id="HE-2",
        source="humaneval",
        prompt="""Write a Python function called `separate_paren_groups` that takes a string
containing multiple groups of nested parentheses. Separate those groups into
separate strings and return a list. Each group is balanced (has equal open/close
parens) and not nested within each other. Ignore spaces in the input.

Example:
    separate_paren_groups('( ) (( )) (( )( ))') -> ['()', '(())', '(()())']""",
        canonical_solution="""def separate_paren_groups(paren_string):
    result = []
    current_group = ''
    depth = 0
    for char in paren_string:
        if char == '(':
            depth += 1
            current_group += char
        elif char == ')':
            depth -= 1
            current_group += char
            if depth == 0:
                result.append(current_group)
                current_group = ''
    return result""",
        test_cases=[
            "assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']",
            "assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']",
        ],
        entry_point="separate_paren_groups",
    ),
    CodeTask(
        task_id="HE-3",
        source="humaneval",
        prompt="""Write a Python function called `truncate_number` that takes a positive
floating point number and returns the decimal part (fractional part) of it.

Example:
    truncate_number(3.5) -> 0.5""",
        canonical_solution="""def truncate_number(number):
    return number % 1.0""",
        test_cases=[
            "assert truncate_number(3.5) == 0.5",
            "assert abs(truncate_number(1.33) - 0.33) < 1e-6",
            "assert abs(truncate_number(123.456) - 0.456) < 1e-6",
        ],
        entry_point="truncate_number",
    ),
    CodeTask(
        task_id="HE-4",
        source="humaneval",
        prompt="""Write a Python function `below_zero` that takes a list of deposit/withdrawal
operations on a bank account starting at zero. Return True if the balance
goes below zero at any point, otherwise return False.

Example:
    below_zero([1, 2, 3]) -> False
    below_zero([1, 2, -4, 5]) -> True""",
        canonical_solution="""def below_zero(operations):
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False""",
        test_cases=[
            "assert below_zero([]) == False",
            "assert below_zero([1, 2, -3, 1, 2, -3]) == False",
            "assert below_zero([1, 2, -4, 5, 6]) == True",
        ],
        entry_point="below_zero",
    ),
    CodeTask(
        task_id="HE-5",
        source="humaneval",
        prompt="""Write a Python function `mean_absolute_deviation` that takes a list of numbers
and returns the Mean Absolute Deviation (MAD) around the mean of the dataset.

MAD = mean(|x - mean(x)|) for all x in the list.

Example:
    mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) -> 1.0""",
        canonical_solution="""def mean_absolute_deviation(numbers):
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)""",
        test_cases=[
            "assert mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) == 1.0",
            "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6",
        ],
        entry_point="mean_absolute_deviation",
    ),
    CodeTask(
        task_id="MBPP-1",
        source="mbpp",
        prompt="""Write a Python function `find_max_sum` that finds the maximum sum of a
contiguous subarray within a given list of integers (Kadane's algorithm).

Example:
    find_max_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) -> 6""",
        canonical_solution="""def find_max_sum(arr):
    max_sum = arr[0]
    current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum""",
        test_cases=[
            "assert find_max_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6",
            "assert find_max_sum([1]) == 1",
            "assert find_max_sum([-1, -2, -3]) == -1",
        ],
        entry_point="find_max_sum",
    ),
    CodeTask(
        task_id="MBPP-2",
        source="mbpp",
        prompt="""Write a Python function `is_palindrome` that checks whether a given string
is a palindrome, ignoring case and non-alphanumeric characters.

Example:
    is_palindrome('A man a plan a canal Panama') -> True
    is_palindrome('race a car') -> False""",
        canonical_solution="""def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]""",
        test_cases=[
            "assert is_palindrome('A man a plan a canal Panama') == True",
            "assert is_palindrome('race a car') == False",
            "assert is_palindrome('') == True",
            "assert is_palindrome('Was it a car or a cat I saw') == True",
        ],
        entry_point="is_palindrome",
    ),
    CodeTask(
        task_id="MBPP-3",
        source="mbpp",
        prompt="""Write a Python function `flatten_list` that takes a nested list (of arbitrary
depth) and returns a flat list of all elements.

Example:
    flatten_list([1, [2, [3, [4]], 5]]) -> [1, 2, 3, 4, 5]""",
        canonical_solution="""def flatten_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result""",
        test_cases=[
            "assert flatten_list([1, [2, [3, [4]], 5]]) == [1, 2, 3, 4, 5]",
            "assert flatten_list([]) == []",
            "assert flatten_list([1, 2, 3]) == [1, 2, 3]",
        ],
        entry_point="flatten_list",
    ),
    CodeTask(
        task_id="MBPP-4",
        source="mbpp",
        prompt="""Write a Python function `count_words` that takes a sentence (string) and
returns a dictionary with the frequency count of each word (case-insensitive).

Example:
    count_words('Hello world hello') -> {'hello': 2, 'world': 1}""",
        canonical_solution="""def count_words(sentence):
    words = sentence.lower().split()
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts""",
        test_cases=[
            "assert count_words('Hello world hello') == {'hello': 2, 'world': 1}",
            "assert count_words('') == {}",
            "assert count_words('a a a') == {'a': 3}",
        ],
        entry_point="count_words",
    ),
    CodeTask(
        task_id="MBPP-5",
        source="mbpp",
        prompt="""Write a Python function `rotate_list` that rotates a list to the right by k
positions. Elements that fall off the end wrap around to the front.

Example:
    rotate_list([1, 2, 3, 4, 5], 2) -> [4, 5, 1, 2, 3]""",
        canonical_solution="""def rotate_list(lst, k):
    if not lst:
        return lst
    k = k % len(lst)
    return lst[-k:] + lst[:-k] if k else lst[:]""",
        test_cases=[
            "assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]",
            "assert rotate_list([1, 2, 3], 0) == [1, 2, 3]",
            "assert rotate_list([], 3) == []",
            "assert rotate_list([1, 2, 3], 3) == [1, 2, 3]",
        ],
        entry_point="rotate_list",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# DATASET LOADER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Loads tasks for experiments.

    Priority:
      1. Built-in mini dataset (always works, no downloads)
      2. HuggingFace `datasets` library (if installed + internet)
    """

    def load_builtin(self, n: Optional[int] = None) -> List[CodeTask]:
        tasks = BUILTIN_TASKS[:n] if n else BUILTIN_TASKS
        print(f"[DatasetLoader] Loaded {len(tasks)} built-in tasks.")
        return tasks

    def load_humaneval_hf(self, n: Optional[int] = None) -> List[CodeTask]:
        """Load from HuggingFace (requires: pip install datasets)"""
        try:
            from datasets import load_dataset
            ds = load_dataset("openai_humaneval", split="test")
            tasks = []
            for row in ds:
                tasks.append(CodeTask(
                    task_id=row["task_id"],
                    source="humaneval",
                    prompt=row["prompt"],
                    canonical_solution=row["canonical_solution"],
                    test_cases=[row["test"]],
                    entry_point=row["entry_point"],
                ))
                if n and len(tasks) >= n:
                    break
            print(f"[DatasetLoader] Loaded {len(tasks)} HumanEval tasks from HuggingFace.")
            return tasks
        except ImportError:
            print("[DatasetLoader] `datasets` not installed. Run: pip install datasets")
            return []
        except Exception as e:
            print(f"[DatasetLoader] HuggingFace load failed: {e}")
            return []

    def load_mbpp_hf(self, n: Optional[int] = None) -> List[CodeTask]:
        """Load from HuggingFace (requires: pip install datasets)"""
        try:
            from datasets import load_dataset
            ds = load_dataset("mbpp", split="test")
            tasks = []
            for row in ds:
                tasks.append(CodeTask(
                    task_id=f"MBPP-{row['task_id']}",
                    source="mbpp",
                    prompt=row["text"],
                    canonical_solution=row["code"],
                    test_cases=row["test_list"],
                    entry_point="",
                ))
                if n and len(tasks) >= n:
                    break
            print(f"[DatasetLoader] Loaded {len(tasks)} MBPP tasks from HuggingFace.")
            return tasks
        except ImportError:
            print("[DatasetLoader] `datasets` not installed. Run: pip install datasets")
            return []
        except Exception as e:
            print(f"[DatasetLoader] HuggingFace load failed: {e}")
            return []

    def load_all(self, n_each: int = 5) -> List[CodeTask]:
        tasks = []
        sources_used = []
    
        # Try builtin first
        builtin_tasks = self.load_builtin(n_each)
        tasks += builtin_tasks
        sources_used.append(f"builtin ({len(builtin_tasks)})")
    
        # Try HumanEval if available
        try:
            hf_tasks = self.load_humaneval_hf(n_each)
            if hf_tasks:
                tasks += hf_tasks
                sources_used.append(f"HumanEval HF ({len(hf_tasks)})")
        except Exception as e:
            print(f"[DatasetLoader] HumanEval HF failed: {e}")
        
        # Try MBPP if available
        try:
            mbpp_tasks = self.load_mbpp_hf(n_each)
            if mbpp_tasks:
                tasks += mbpp_tasks
                sources_used.append(f"MBPP HF ({len(mbpp_tasks)})")
        except Exception as e:
            print(f"[DatasetLoader] MBPP HF failed: {e}")
        
        # Return up to n_each*3 tasks (built-in + humaneval + mbpp)
        result = tasks[:n_each*3]
        print(f"[DatasetLoader] ✓ Loaded {len(result)} total tasks from: {', '.join(sources_used)}")
        return result