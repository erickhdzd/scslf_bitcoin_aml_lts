"""Export node embeddings from the best full-graph GraphSAGE checkpoints.

The exported representation is the final GraphSAGE hidden state immediately
before the MLP classifier. With the default training configuration its shape is
[num_nodes, 192]. One file is generated for each seed.

Example:
    python src/export_graphsage_embeddings.py \
        --seeds 42 43 44 45 46 \
        --checkpoint-root results/graphsage \
        --output-dir data/processed/graphsage_embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from artifact_utils import portable_path
from train_graphsage import (
    CausalGraphSAGE,
    load_graph,
    resolve_device,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export pre-classifier embeddings from trained GraphSAGE models."
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("data/processed/graphsage"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("results/graphsage"),
        help="Directory containing seed_<seed>/best_model.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/graphsage_embeddings"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Data type used only for the saved embedding tensors.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def checkpoint_architecture(checkpoint: dict[str, Any]) -> dict[str, Any]:
    configuration = checkpoint.get("configuration")
    if not isinstance(configuration, dict):
        raise TypeError("Checkpoint does not contain a valid configuration dictionary.")

    required = ("hidden_dim", "num_layers", "classifier_hidden_dim", "dropout")
    missing = [name for name in required if name not in configuration]
    if missing:
        raise ValueError(f"Checkpoint configuration is missing: {missing}")

    return {
        "hidden_dim": int(configuration["hidden_dim"]),
        "num_layers": int(configuration["num_layers"]),
        "classifier_hidden_dim": int(configuration["classifier_hidden_dim"]),
        "dropout": float(configuration["dropout"]),
    }


@torch.inference_mode()
def encode_nodes(model: CausalGraphSAGE, x: Tensor, adjacency: Tensor) -> Tensor:
    """Return the exact hidden state consumed by ``model.classifier``."""
    model.eval()
    hidden = x
    for layer in model.layers:
        hidden = layer(hidden, adjacency)
        hidden = torch.nn.functional.gelu(hidden)
        hidden = model.dropout(hidden)
    return hidden


@torch.inference_mode()
def verify_embeddings(
    model: CausalGraphSAGE,
    x: Tensor,
    adjacency: Tensor,
    embeddings: Tensor,
) -> float:
    """Verify that exported embeddings reproduce the model logits."""
    direct_logits = model(x, adjacency)
    reconstructed_logits = model.classifier(embeddings).squeeze(-1)
    maximum_error = torch.max(torch.abs(direct_logits - reconstructed_logits))
    return float(maximum_error.item())


def device_to_string(
    device: torch.device,
) -> str:
    """
    Return a stable representation of the device in use.
    """

    if device.type == "cuda":
        index = 0 if device.index is None else int(device.index)

        return f"cuda:{index}"

    return "cpu"


def main() -> int:
    args = parse_args()

    graph_dir = args.graph_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    device_label = device_to_string(device)
    data, graph_metadata = load_graph(
        graph_dir,
        device,
    )

    # These shared arrays define the exact row-to-transaction correspondence.
    np.save(output_dir / "graph_ids.npy", data.ids)
    np.save(output_dir / "graph_times.npy", data.times)

    storage_dtype = torch.float32 if args.storage_dtype == "float32" else torch.float16

    manifest: dict[str, Any] = {
        "graph_dir": portable_path(graph_dir),
        "checkpoint_root": portable_path(checkpoint_root),
        "device": device_label,
        "storage_dtype": args.storage_dtype,
        "node_count": int(data.x.shape[0]),
        "input_dimension": int(data.x.shape[1]),
        "graph_metadata": graph_metadata,
        "seeds": {},
    }

    print("=" * 88)
    print("EXPORT GRAPHSAGE NODE EMBEDDINGS")
    print("=" * 88)
    print(f"Device          : {device_label}")
    print(f"Nodes           : {data.x.shape[0]:,}")
    print(f"Input features  : {data.x.shape[1]:,}")
    print(f"Output directory: {output_dir}")

    for seed in args.seeds:
        set_global_seed(seed)
        checkpoint_path = checkpoint_root / f"seed_{seed}" / "best_model.pt"
        require_file(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        architecture = checkpoint_architecture(checkpoint)

        model = CausalGraphSAGE(
            input_dim=int(data.x.shape[1]),
            hidden_dim=architecture["hidden_dim"],
            num_layers=architecture["num_layers"],
            classifier_hidden_dim=architecture["classifier_hidden_dim"],
            dropout=architecture["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        embeddings = encode_nodes(model, data.x, data.adjacency)
        if embeddings.ndim != 2 or embeddings.shape[0] != data.x.shape[0]:
            raise RuntimeError(
                f"Unexpected embedding shape for seed {seed}: {tuple(embeddings.shape)}"
            )
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError(f"Non-finite embeddings found for seed {seed}.")

        maximum_logit_error = verify_embeddings(
            model=model,
            x=data.x,
            adjacency=data.adjacency,
            embeddings=embeddings,
        )
        if maximum_logit_error > 1e-5:
            raise RuntimeError(
                "Embedding verification failed for seed "
                f"{seed}: maximum logit error={maximum_logit_error:.8e}"
            )

        output_path = output_dir / f"graphsage_embeddings_seed{seed}.pt"
        cpu_embeddings = (
            embeddings.detach()
            .to(
                device="cpu",
                dtype=storage_dtype,
            )
            .contiguous()
        )
        torch.save(cpu_embeddings, output_path)

        seed_metadata = {
            "seed": int(seed),
            "checkpoint": portable_path(checkpoint_path),
            "best_epoch": int(checkpoint["epoch"]),
            "validation_threshold": float(checkpoint["threshold"]),
            "shape": list(cpu_embeddings.shape),
            "dtype": str(cpu_embeddings.dtype),
            "maximum_logit_reconstruction_error": maximum_logit_error,
            "architecture": architecture,
            "output_file": output_path.name,
        }
        manifest["seeds"][str(seed)] = seed_metadata

        print("-" * 88)
        print(f"Seed            : {seed}")
        print(f"Best epoch      : {checkpoint['epoch']}")
        print(f"Embedding shape : {tuple(cpu_embeddings.shape)}")
        print(f"Saved dtype     : {cpu_embeddings.dtype}")
        print(f"Logit max error : {maximum_logit_error:.3e}")
        print(f"Saved in        : {portable_path(output_path)}")

        del model, embeddings, cpu_embeddings, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(output_dir / "embedding_manifest.json", manifest)
    print("=" * 88)
    print(f"Manifest saved in: {output_dir / 'embedding_manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nExport interrupted by the user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
