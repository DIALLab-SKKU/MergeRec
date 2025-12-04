import json
from pathlib import Path

from ..dataset import RecItemDataset, RecDataset


def load_json_files(dataset_path: Path, max_items: int):
    with open(dataset_path / "train.json", "r") as f:
        train_seq = {int(k): v for k, v in json.load(f).items()}
    with open(dataset_path / "val.json", "r") as f:
        val_seq = {int(k): v for k, v in json.load(f).items()}
    with open(dataset_path / "test.json", "r") as f:
        test_seq = {int(k): v for k, v in json.load(f).items()}

    with open(dataset_path / "meta_data.json", "r") as f:
        metadata = json.load(f)

    with open(dataset_path / "umap.json", "r") as f:
        umap = json.load(f)
    with open(dataset_path / "smap.json", "r") as f:
        smap = json.load(f)

    for k, v in val_seq.items():
        val_seq[k] = train_seq.get(k, []) + v
    for k, v in test_seq.items():
        test_seq[k] = val_seq.get(k, []) + v

    # Create dataset
    item_dataset = RecItemDataset(list(smap.values()))
    train_dataset = RecDataset(train_seq, sample=False, max_items=max_items)
    val_dataset = RecDataset(val_seq, sample=False, max_items=max_items)
    test_dataset = RecDataset(test_seq, sample=False, max_items=max_items)

    # Map metadata to item ids
    metadata = {smap[item_asin]: item_metadata for item_asin, item_metadata in metadata.items() if item_asin in smap}

    return item_dataset, train_dataset, val_dataset, test_dataset, metadata, umap, smap
