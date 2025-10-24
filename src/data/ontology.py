"""
Ontology and label mapping system for music structure recognition.

Defines canonical labels and mappings from various dataset-specific labels
to the standard ontology.
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import json


@dataclass
class LabelMapping:
    """Configuration for label mapping."""

    canonical_labels: List[str]
    mappings: Dict[str, str]
    dataset_specific: Dict[str, Dict[str, str]]


# Canonical labels
CANONICAL_LABELS = [
    "INTRO",
    "VERSE",
    "PRE",
    "CHORUS",
    "BRIDGE",
    "SOLO",
    "OUTRO",
    "OTHER",
]

# General mappings
GENERAL_MAPPINGS = {
    # General
    "intro": "INTRO",
    "interlude": "OTHER",
    "break": "OTHER",
    "drop": "CHORUS",
    "hook": "CHORUS",
    "refrain": "CHORUS",
    "build": "PRE",
    "rise": "PRE",
    "verse": "VERSE",
    "pre-chorus": "PRE",
    "prechorus": "PRE",
    "chorus": "CHORUS",
    "bridge": "BRIDGE",
    "solo": "SOLO",
    "outro": "OUTRO",
    "ending": "OUTRO",
    "coda": "OUTRO",
    # Dataset-specific synonyms
    "rap": "VERSE",
    "vocal": "OTHER",
    "instrumental": "OTHER",
}

# Dataset-specific mappings (extend as needed)
DATASET_SPECIFIC_MAPPINGS = {
    "harmonix": {
        "instrumental": "OTHER",
        "silence": "OTHER",
    },
    "salami": {
        "silence": "OTHER",
        "speech": "OTHER",
    },
    "beatles": {
        "instrumental": "OTHER",
    },
    "spam": {
        "silence": "OTHER",
    },
    "ccmusic": {
        "间奏": "OTHER",  # Chinese for interlude
        "前奏": "INTRO",  # Chinese for intro
        "主歌": "VERSE",  # Chinese for verse
        "副歌": "CHORUS",  # Chinese for chorus
        "verse a": "VERSE",
        "verse b": "VERSE",
        "verse c": "VERSE",
        "verse d": "VERSE",
        "verse e": "VERSE",
        "verse": "VERSE",
        "chorus a": "CHORUS",
        "chorus b": "CHORUS",
        "chorus c": "CHORUS",
        "chorus d": "CHORUS",
        "chorus e": "CHORUS",
        "chorus f": "CHORUS",
        "chorus g": "CHORUS",
        "pre chorus": "PRE",
        "pre chorus a": "PRE",
        "pre chorus b": "PRE",
        "pre chorus c": "PRE",
        "re intro": "INTRO",
        "reintro": "INTRO",
        "re intro a": "INTRO",
        "re intro b": "INTRO",
        "bridge 1": "BRIDGE",
        "bridge 2": "BRIDGE",
        "interlude a": "OTHER",
        "interlude b": "OTHER",
    },
}


class LabelMapper:
    """Handles mapping of labels from various datasets to canonical labels."""

    def __init__(self, config: Optional[LabelMapping] = None):
        if config is None:
            config = LabelMapping(
                canonical_labels=CANONICAL_LABELS,
                mappings=GENERAL_MAPPINGS,
                dataset_specific=DATASET_SPECIFIC_MAPPINGS,
            )
        self.config = config
        self._build_mapping_cache()

    def _build_mapping_cache(self):
        """Build a fast lookup cache for all mappings."""
        self._cache = {}

        # Add general mappings
        for key, value in self.config.mappings.items():
            lowered = key.lower()
            normalized = self._normalize_label(lowered)
            self._cache[lowered] = value
            self._cache[normalized] = value

        # Add dataset-specific mappings
        for dataset, mappings in self.config.dataset_specific.items():
            for key, value in mappings.items():
                lowered = key.lower()
                normalized = self._normalize_label(lowered)
                for variant in {lowered, normalized}:
                    cache_key = f"{dataset}:{variant}"
                    self._cache[cache_key] = value

    def map_label(self, label: str, dataset: Optional[str] = None) -> str:
        """
        Map a label to canonical form.

        Args:
            label: Input label to map
            dataset: Optional dataset name for dataset-specific mapping

        Returns:
            Canonical label
        """
        if not label or not isinstance(label, str):
            return "OTHER"

        label_lower = label.lower().strip()

        # Try dataset-specific mapping first
        if dataset:
            normalized_dataset_label = self._normalize_label(label_lower)
            dataset_candidates = {label_lower, normalized_dataset_label}
            dataset_candidates.update(
                self._generate_partial_candidates(normalized_dataset_label)
            )
            for candidate in dataset_candidates:
                cache_key = f"{dataset}:{candidate}"
                if cache_key in self._cache:
                    return self._cache[cache_key]

        # Try general mapping
        normalized_label = self._normalize_label(label_lower)
        candidates = {label_lower, normalized_label}
        candidates.update(self._generate_partial_candidates(normalized_label))
        for candidate in candidates:
            if candidate in self._cache:
                return self._cache[candidate]

        # Check if it's already canonical
        if label.upper() in self.config.canonical_labels:
            return label.upper()

        # Default to OTHER for unknown labels
        return "OTHER"

    def map_labels(self, labels: List[str], dataset: Optional[str] = None) -> List[str]:
        """Map a list of labels to canonical form."""
        return [self.map_label(label, dataset) for label in labels]

    def get_canonical_labels(self) -> List[str]:
        """Get the list of canonical labels."""
        return self.config.canonical_labels.copy()

    def get_label_stats(
        self, labels: List[str], dataset: Optional[str] = None
    ) -> Dict[str, int]:
        """Get statistics about label distribution after mapping."""
        mapped_labels = self.map_labels(labels, dataset)
        stats = {}
        for label in mapped_labels:
            stats[label] = stats.get(label, 0) + 1
        return stats

    def save_config(self, filepath: str):
        """Save the mapping configuration to a JSON file."""
        config_dict = {
            "canonical_labels": self.config.canonical_labels,
            "mappings": self.config.mappings,
            "dataset_specific": self.config.dataset_specific,
        }
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(cls, filepath: str) -> "LabelMapper":
        """Load mapping configuration from a JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)

        config = LabelMapping(
            canonical_labels=config_dict["canonical_labels"],
            mappings=config_dict["mappings"],
            dataset_specific=config_dict["dataset_specific"],
        )
        return cls(config)

    def _normalize_label(self, label: str) -> str:
        """
        Normalize a label string for matching.

        - Lowercase
        - Replace hyphens/underscores with spaces
        - Collapse whitespace
        - Remove trailing single letters/numbers
        """
        normalized = label.replace("_", " ").replace("-", " ")
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Remove trailing single letter or digit tokens (e.g., "verse a" -> "verse")
        normalized = re.sub(r"\s+[a-z0-9]$", "", normalized)

        return normalized

    def _generate_partial_candidates(self, normalized_label: str) -> List[str]:
        """Generate partial label candidates (first token(s))."""
        tokens = normalized_label.split()
        candidates: List[str] = []

        if not tokens:
            return candidates

        # First two tokens (useful for 'pre chorus')
        if len(tokens) >= 2:
            candidates.append(" ".join(tokens[:2]))

        # First token alone
        candidates.append(tokens[0])

        return candidates


# Global instance for easy access
label_mapper = LabelMapper()


def map_label(label: str, dataset: Optional[str] = None) -> str:
    """Convenience function for mapping a single label."""
    return label_mapper.map_label(label, dataset)


def map_labels(labels: List[str], dataset: Optional[str] = None) -> List[str]:
    """Convenience function for mapping a list of labels."""
    return label_mapper.map_labels(labels, dataset)


if __name__ == "__main__":
    # Test the mapping system
    test_labels = [
        "intro",
        "verse",
        "pre-chorus",
        "chorus",
        "bridge",
        "solo",
        "outro",
        "instrumental",
        "silence",
        "unknown_label",
        "INTRO",
        "VERSE",
    ]

    print("Testing label mapping:")
    for label in test_labels:
        mapped = map_label(label)
        print(f"  {label:15} -> {mapped}")

    print(f"\nCanonical labels: {CANONICAL_LABELS}")

    # Test dataset-specific mapping
    print("\nTesting dataset-specific mapping:")
    chinese_labels = ["前奏", "主歌", "副歌", "间奏"]
    for label in chinese_labels:
        mapped = map_label(label, "ccmusic")
        print(f"  {label:10} (ccmusic) -> {mapped}")
