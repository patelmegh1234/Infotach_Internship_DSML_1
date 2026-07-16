# Shubhangi, 2026-07-13.
from pathlib import Path

import kagglehub


def main() -> None:
    dataset = "awsaf49/asvpoof-2019-dataset"
    path = kagglehub.dataset_download(dataset)
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloaded {dataset}")
    print(f"Path to dataset files: {path}")
    print(f"Recommended local working folder: {data_dir.resolve()}")


if __name__ == "__main__":
    main()

