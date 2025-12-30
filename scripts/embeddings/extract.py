from pathlib import Path
from sys import argv

import torch

if len(argv) != 2:
    print("Usage: python extract.py <model_checkpoint>")
    exit(1)

model_checkpoint = Path(argv[1])

model_ckpt = torch.load(model_checkpoint, map_location="cpu")
embedding = model_ckpt["item_embeddings"]
output_path = model_checkpoint.parent / f"item_embedding.pt"

torch.save(embedding, output_path)
print(f"Saved item embeddings to {output_path}")
