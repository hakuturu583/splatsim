#!/usr/bin/env python3
"""Generate gRPC/protobuf Python code from .proto files.

Usage:
    python scripts/generate_proto.py

The generated files are written to src/splatsim/grpc_service/_generated/.
Import paths are automatically fixed for the flat _generated layout.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "proto"
OUT_DIR = ROOT / "src" / "splatsim" / "grpc_service" / "_generated"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    proto_files = sorted(PROTO_DIR.rglob("*.proto"))
    if not proto_files:
        print("No .proto files found under", PROTO_DIR, file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        f"--pyi_out={OUT_DIR}",
        *[str(p) for p in proto_files],
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    # protoc generates files under nested package dirs (e.g. _generated/splatsim/v1/).
    # Move them to the flat _generated/ directory and fix import paths.
    _flatten_and_fix_imports()

    # Ensure __init__.py exists
    init_py = OUT_DIR / "__init__.py"
    if not init_py.exists():
        init_py.touch()

    print("Generated files in", OUT_DIR)


def _flatten_and_fix_imports() -> None:
    """Move files from nested subdirs to OUT_DIR and rewrite imports."""
    # Find all generated .py/.pyi files in nested subdirs
    nested_files = [
        f
        for f in OUT_DIR.rglob("*.py*")
        if f.parent != OUT_DIR and f.suffix in (".py", ".pyi")
    ]

    for src in nested_files:
        dst = OUT_DIR / src.name
        dst.write_text(src.read_text())
        src.unlink()

    # Remove empty nested package directories
    for d in sorted(OUT_DIR.rglob("*"), reverse=True):
        if d.is_dir() and d != OUT_DIR:
            try:
                d.rmdir()
            except OSError:
                pass

    # Fix import paths: the proto package (e.g. "splatsim.v1") → flat _generated
    for py_file in OUT_DIR.glob("*.py"):
        text = py_file.read_text()
        fixed = re.sub(
            r"from splatsim\.v1 import",
            "from splatsim.grpc_service._generated import",
            text,
        )
        if fixed != text:
            py_file.write_text(fixed)
            print(f"  Fixed imports in {py_file.name}")


if __name__ == "__main__":
    main()
