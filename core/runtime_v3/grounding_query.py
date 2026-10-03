"""Generic linguistic normalization at the SAM grounding interface."""

from __future__ import annotations


class GroundingQueryNormalizer:
    """Normalize surface form while leaving the semantic TaskSpec untouched."""

    LEADING_DETERMINERS = frozenset({"the", "a", "an"})

    def normalize(self, semantic_phrase: str) -> str:
        words = " ".join(str(semantic_phrase).split()).casefold().split(" ")
        if len(words) > 1 and words[0] in self.LEADING_DETERMINERS:
            words = words[1:]
        return " ".join(words)
