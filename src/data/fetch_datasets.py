"""
Web fetcher for downloading datasets and annotations from various sources.

Handles Hugging Face datasets, Git repositories, and project pages.
Keeps secrets in environment variables.
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

import requests
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    source: str  # 'hf', 'git', 'url'
    repo_id: Optional[str] = None
    url: Optional[str] = None
    files: Optional[List[str]] = None
    audio_source: Optional[str] = None  # 'provided', 'youtube', 'external'
    git_branch: Optional[str] = None


# Dataset configurations
DATASET_CONFIGS = {
    "harmonix": DatasetConfig(
        name="harmonix",
        source="hf",
        repo_id="m-a-p/HarmonixSet",
        files=["annotations.json", "metadata.json"],
        audio_source="provided",
    ),
    "salami": DatasetConfig(
        name="salami",
        source="git",
        url="https://github.com/DDMAL/salami-data-public.git",
        files=["annotations/*.txt"],
        audio_source="external",
    ),
    "beatles": DatasetConfig(
        name="beatles",
        source="hf",
        repo_id="m-a-p/beatles_structure_annotations",
        files=["annotations.json"],
        audio_source="external",
    ),
    "spam": DatasetConfig(
        name="spam",
        source="git",
        url="https://github.com/urinieto/SPAM.git",
        files=["annotations/*.txt"],
        audio_source="external",
    ),
    "ccmusic": DatasetConfig(
        name="ccmusic",
        source="hf",
        repo_id="ccmusic-database/song_structure",
        files=["train.json", "validation.json", "test.json"],
        audio_source="external",
    ),
}


class DatasetFetcher:
    """Handles downloading datasets from various sources."""

    def __init__(self, data_dir: str = "data/raw", hf_token: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if not self.hf_token:
            logger.warning(
                "No Hugging Face token provided. Some datasets may not be accessible."
            )

    def fetch_dataset(self, dataset_name: str, force_download: bool = False) -> Path:
        """
        Fetch a dataset by name.

        Args:
            dataset_name: Name of the dataset to fetch
            force_download: Whether to force re-download even if exists

        Returns:
            Path to the downloaded dataset directory
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = DATASET_CONFIGS[dataset_name]
        dataset_dir = self.data_dir / dataset_name

        if dataset_dir.exists() and not force_download:
            logger.info(f"Dataset {dataset_name} already exists at {dataset_dir}")
            return dataset_dir

        logger.info(f"Fetching dataset {dataset_name} from {config.source}")

        if config.source == "hf":
            self._fetch_huggingface(config, dataset_dir, force_download)
        elif config.source == "git":
            self._fetch_git(config, dataset_dir, force_download)
        elif config.source == "url":
            self._fetch_url(config, dataset_dir)
        else:
            raise ValueError(f"Unknown source: {config.source}")

        return dataset_dir

    def _fetch_huggingface(
        self, config: DatasetConfig, dataset_dir: Path, force_download: bool
    ):
        """Fetch dataset from Hugging Face."""
        try:
            snapshot_path = snapshot_download(
                repo_id=config.repo_id,
                repo_type="dataset",
                token=self.hf_token,
                local_dir=None,
                local_dir_use_symlinks=False,
                force_download=force_download,
            )
            logger.info(f"Snapshot downloaded to {snapshot_path}")

            if dataset_dir.exists():
                if force_download:
                    shutil.rmtree(dataset_dir)
                elif any(dataset_dir.iterdir()):
                    logger.info(
                        "Target directory already contains files; skipping copy."
                    )
                    return

            shutil.copytree(snapshot_path, dataset_dir, dirs_exist_ok=True)
            logger.info(f"Copied snapshot contents into {dataset_dir}")

        except Exception as e:
            logger.error(f"Failed to fetch Hugging Face dataset {config.repo_id}: {e}")
            raise

    def _fetch_git(
        self, config: DatasetConfig, dataset_dir: Path, force_download: bool
    ):
        """Fetch dataset from Git repository."""
        if not config.url:
            raise ValueError("Git URL not provided")

        try:
            if dataset_dir.exists():
                if force_download:
                    shutil.rmtree(dataset_dir)
                else:
                    logger.info(f"Git dataset already present at {dataset_dir}")
                    return

            clone_cmd = ["git", "clone", config.url, str(dataset_dir)]
            if config.git_branch:
                clone_cmd.extend(["--branch", config.git_branch])

            subprocess.run(clone_cmd, check=True, capture_output=True)
            logger.info(f"Cloned repository to {dataset_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise

    def _fetch_url(self, config: DatasetConfig, dataset_dir: Path):
        """Fetch dataset from URL (for direct downloads)."""
        if not config.url:
            raise ValueError("URL not provided")

        dataset_dir.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(config.url, stream=True)
            response.raise_for_status()

            # For now, just create a placeholder file
            # In practice, you'd need to handle different URL types
            placeholder_file = dataset_dir / "README.md"
            with open(placeholder_file, "w") as f:
                f.write(f"Dataset source: {config.url}\n")
                f.write("Manual download required.\n")

            logger.info(f"Created placeholder for URL dataset at {dataset_dir}")

        except Exception as e:
            logger.error(f"Failed to fetch URL dataset: {e}")
            raise

    def fetch_all_datasets(self, force_download: bool = False) -> Dict[str, Path]:
        """Fetch all configured datasets."""
        results = {}
        for dataset_name in DATASET_CONFIGS:
            try:
                results[dataset_name] = self.fetch_dataset(
                    dataset_name, force_download
                )
            except Exception as e:
                logger.error(f"Failed to fetch {dataset_name}: {e}")
                results[dataset_name] = None
        return results

    def list_available_datasets(self) -> List[str]:
        """List all available dataset names."""
        return list(DATASET_CONFIGS.keys())

    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetConfig]:
        """Get configuration for a dataset."""
        return DATASET_CONFIGS.get(dataset_name)


def main():
    """CLI interface for dataset fetching."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch music structure datasets")
    parser.add_argument("--dataset", help="Specific dataset to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--data-dir", default="data/raw", help="Data directory")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    fetcher = DatasetFetcher(data_dir=args.data_dir)

    if args.all:
        results = fetcher.fetch_all_datasets(force_download=args.force)
        print("Fetch results:")
        for name, path in results.items():
            status = "✓" if path else "✗"
            print(f"  {status} {name}: {path}")
    elif args.dataset:
        try:
            path = fetcher.fetch_dataset(args.dataset, force_download=args.force)
            print(f"✓ Fetched {args.dataset} to {path}")
        except Exception as e:
            print(f"✗ Failed to fetch {args.dataset}: {e}")
    else:
        print("Available datasets:")
        for name in fetcher.list_available_datasets():
            config = fetcher.get_dataset_info(name)
            print(f"  {name}: {config.source} ({config.repo_id or config.url})")


if __name__ == "__main__":
    main()
