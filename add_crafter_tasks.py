"""Add Crafter achievement embeddings to the DFM server vocabulary.

Encodes the 22 Crafter achievements using Qwen3-Embedding-8B and appends
them to an existing embeddings.pt file.

Usage:
    python add_crafter_tasks.py --embeddings checkpoint/embeddings.pt
"""

import argparse
import shutil
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

# 22 Crafter achievements (same order as CrafterGC ACHIEVEMENTS)
CRAFTER_ACHIEVEMENTS = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]


def main():
    parser = argparse.ArgumentParser(description="Add Crafter tasks to DFM embeddings")
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to existing embeddings.pt file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="Sentence-transformers model name",
    )
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    # Load existing embeddings
    print(f"Loading existing embeddings from {emb_path}...")
    data = torch.load(emb_path, map_location="cpu", weights_only=True)
    existing_keys = set(data["keys"])
    print(f"  Existing vocabulary: {len(data['keys']):,} tasks")

    # Determine which tasks need encoding
    task_strings = [a.replace("_", " ") for a in CRAFTER_ACHIEVEMENTS]
    new_tasks = [t for t in task_strings if t not in existing_keys]

    if not new_tasks:
        print("All Crafter tasks already in vocabulary. Nothing to do.")
        return

    print(f"  New tasks to encode: {len(new_tasks)}")
    for t in new_tasks:
        print(f"    - {t}")

    # Encode new tasks
    print(f"\nLoading model {args.model}...")
    model = SentenceTransformer(
        args.model,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    print(f"Encoding {len(new_tasks)} tasks...")
    embeddings = model.encode(
        new_tasks,
        batch_size=len(new_tasks),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    new_embs = torch.from_numpy(embeddings).to(torch.bfloat16)
    print(f"  Embedding shape: {new_embs.shape}")

    # Verify dimensions match
    assert new_embs.shape[1] == data["embeddings"].shape[1], (
        f"Dimension mismatch: new={new_embs.shape[1]}, existing={data['embeddings'].shape[1]}"
    )

    # Backup original file
    backup_path = emb_path.with_suffix(".pt.bak")
    print(f"\nBacking up original to {backup_path}")
    shutil.copy2(emb_path, backup_path)

    # Merge and save
    merged_keys = data["keys"] + new_tasks
    merged_embs = torch.cat([data["embeddings"], new_embs], dim=0)

    torch.save({"keys": merged_keys, "embeddings": merged_embs}, emb_path)
    print(f"Saved {len(merged_keys):,} embeddings to {emb_path}")
    print(f"  Added {len(new_tasks)} Crafter tasks")


if __name__ == "__main__":
    main()
