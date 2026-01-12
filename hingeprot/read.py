#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

def _as_float32(x: float) -> float:
    if np is None:
        return float(x)
    return float(np.float32(x))

def convert_pdb_to_coordinates(in_file="pdb", alpha_out="alpha.cor", coord_out="coordinates") -> int:
    in_path = Path(in_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    selected = []

    with in_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM  "):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            resname = line[17:20].strip()
            chain = (line[21].strip() or " ")
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26] if len(line) > 26 else " "

            try:
                x = _as_float32(float(line[30:38]))
                y = _as_float32(float(line[38:46]))
                z = _as_float32(float(line[46:54]))
            except ValueError:
                continue

            if not selected:
                selected.append((resname, chain, resseq, icode, x, y, z))
                continue

            prev_resname, prev_chain, prev_resseq, prev_icode, *_ = selected[-1]

            if (chain != prev_chain) or (resseq != prev_resseq) or (icode.strip() != ""):
                selected.append((resname, chain, resseq, icode, x, y, z))

    ires = len(selected)

    with Path(alpha_out).open("w", encoding="utf-8") as out:
        for j, (resname, chain, resseq, icode, x, y, z) in enumerate(selected, start=1):
            out.write(
                f"{'ATOM':<4}"
                f"{'':3}"
                f"{j:4d}"
                f"{'':1}"
                f"{'CA':>3}"
                f"{'':2}"
                f"{resname:>3}"
                f"{'':1}"
                f"{chain}"
                f"{resseq:4d}"
                f"{icode}"
                f"{'':3}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
            )

    with Path(coord_out).open("w", encoding="utf-8") as out:
        out.write(f" {ires:4d}\n")
        for j, (resname, chain, resseq, icode, x, y, z) in enumerate(selected, start=1):
            out.write(
                f"{j:5d}"
                f"{x:9.2f}{y:9.2f}{z:9.2f}"
                f"{'':3}"
                f"{resname:>3}"
                f"{'':1}"
                f"{icode}\n"
            )

    return ires

if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else "pdb"
    alpha_out = sys.argv[2] if len(sys.argv) > 2 else "alpha.cor"
    coord_out = sys.argv[3] if len(sys.argv) > 3 else "coordinates"
    n = convert_pdb_to_coordinates(in_file, alpha_out, coord_out)
    print(f"Wrote {alpha_out} and {coord_out} with ires={n}")
