from pathlib import Path
from sys import argv

import torch


def extract_checkpoint(model_checkpoint: Path, output_dir: Path):
    if not model_checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

    if not output_dir.exists():
        print(f"Output directory does not exist. Creating: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = torch.load(model_checkpoint, map_location="cpu")["state_dict"]

    torch.save(state_dict["item_embeddings"], output_dir / f"item_embedding.pt")
    torch.save(state_dict, output_dir / "state_dict.pt")

    print("Extraction complete.")


if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: python extract.py <lightning_checkpoint> <output_dir>")
        exit(1)

    extract_checkpoint(Path(argv[1]), Path(argv[2]))
