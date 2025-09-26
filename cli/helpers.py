from __future__ import annotations

from typing import List, Optional
import logging
import os


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure basic logging for CLI if not already configured.

    Honors ORDERORACLE_LOG_LEVEL env var (default INFO).
    """
    if logging.getLogger().handlers:
        return
    level_name = os.getenv("ORDERORACLE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


def parse_dims_pow2(spec: Optional[str]) -> Optional[List[int]]:
    """Parse a simple powers-of-two spec into a dims schedule.

    Accepted forms:
    - "start-end" or "start:end" → generate [start, start*2, ..., end]
    - "N" (single int) → [N]

    Constraints: start and end must be positive integers, start <= end, and
    each generated value is a power of two progression via doubling.
    """
    if spec is None:
        return None
    s = spec.strip()
    if not s:
        return None
    if "-" in s or ":" in s:
        sep = "-" if "-" in s else ":"
        left, right = s.split(sep, 1)
        try:
            start = int(left.strip())
            end = int(right.strip())
        except ValueError:
            # Typer's BadParameter is referenced only from command modules
            raise ValueError("--dims-pow2 must be in the form START-END, e.g., 16-512")
        if start <= 0 or end <= 0:
            raise ValueError("--dims-pow2 values must be positive")
        if start > end:
            raise ValueError("--dims-pow2 start must be <= end")
        dims: List[int] = []
        val = start
        while val <= end:
            dims.append(val)
            val *= 2
        return dims
    else:
        try:
            single = int(s)
        except ValueError:
            raise ValueError("--dims-pow2 must be an int or range like 16-512")
        if single <= 0:
            raise ValueError("--dims-pow2 must be positive")
        return [single]


def parse_list_option(values: Optional[List[str]], csv: Optional[str]) -> Optional[List[str]]:
    """Merge a repeatable option list with an optional comma/space-separated string.

    Returns None if no values present.
    """
    merged: List[str] = []
    if values:
        merged.extend(values)
    if csv:
        for token in csv.replace(",", " ").split():
            merged.append(token)
    return merged or None


