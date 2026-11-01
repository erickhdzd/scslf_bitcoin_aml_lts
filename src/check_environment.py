from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy
import pandas
import sklearn
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display environment, dependency, CUDA, and GPU diagnostics for the artifact."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    print("=" * 70)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 70)

    print(f"System: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"Python executable name: {Path(sys.executable).name}")
    print(f"NumPy: {numpy.__version__}")
    print(f"Pandas: {pandas.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA bundled with PyTorch: {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")

        properties = torch.cuda.get_device_properties(0)
        memory_gb = properties.total_memory / (1024**3)

        print(f"GPU memory: {memory_gb:.2f} GB")

        x = torch.rand((2000, 2000), device="cuda")
        y = torch.rand((2000, 2000), device="cuda")
        z = x @ y

        print(f"GPU test successful: {z.shape}")
    else:
        print("WARNING: PyTorch is not using the GPU.")

    print("=" * 70)


if __name__ == "__main__":
    main()
