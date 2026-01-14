from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# ------------------------- PDB parsing / formatting -------------------------

@dataclass(frozen=True)
class ResKey:
    chain: str
    resnum: int
    icode: str  # insertion code (single char or space)


def _safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def read_pdb_atoms(pdb_path: str):
    """
    Returns:
      lines: list[str] original lines (without trailing newline)
      atom_idx: list[int] indices into `lines` for ATOM/HETATM records
      atom_keys: list[ResKey] per-atom residue key
      atom_xyz: np.ndarray shape (n_atoms,3)
      ca_keys: list[ResKey] residue keys for CA atoms in order of appearance
      ca_xyz: np.ndarray shape (n_res,3) CA coords in same order as ca_keys
    """
    lines: List[str] = []
    atom_idx: List[int] = []
    atom_keys: List[ResKey] = []
    atom_xyz: List[Tuple[float, float, float]] = []

    ca_keys: List[ResKey] = []
    ca_xyz: List[Tuple[float, float, float]] = []
    seen_ca: set[ResKey] = set()

    with open(pdb_path, "r", errors="ignore") as f:
        for li, raw in enumerate(f):
            line = raw.rstrip("\n")
            lines.append(line)

            rec = line[0:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue

            chain = (line[21] if len(line) > 21 else " ").strip() or "A"
            try:
                resnum = int(line[22:26])
            except Exception:
                continue
            icode = (line[26] if len(line) > 26 else " ")
            key = ResKey(chain=chain, resnum=resnum, icode=icode)

            x = _safe_float(line[30:38])
            y = _safe_float(line[38:46])
            z = _safe_float(line[46:54])

            atom_idx.append(li)
            atom_keys.append(key)
            atom_xyz.append((x, y, z))

            atname = line[12:16].strip() if len(line) >= 16 else ""
            if atname == "CA" and key not in seen_ca:
                seen_ca.add(key)
                ca_keys.append(key)
                ca_xyz.append((x, y, z))

    if not ca_keys:
        raise ValueError("No CA atoms found in the input PDB.")

    return (
        lines,
        atom_idx,
        atom_keys,
        np.asarray(atom_xyz, dtype=float),
        ca_keys,
        np.asarray(ca_xyz, dtype=float),
    )


def format_pdb_line_with_xyz_b(line: str, x: float, y: float, z: float, bfac: float) -> str:
    """
    Update x,y,z and B-factor in an ATOM/HETATM line while preserving the rest.
    """
    if len(line) < 66:
        line = line.ljust(66)

    head = line[:30]
    occ = line[54:60]
    tail = line[66:] if len(line) > 66 else ""

    return f"{head}{x:8.3f}{y:8.3f}{z:8.3f}{occ}{bfac:6.2f}{tail}".rstrip()


# ------------------------- Vector parsing -------------------------

def read_vector_file(vec_path: str, n_res: int) -> np.ndarray:
    """
    Vector file format:
      idx  dx  dy  dz  (optional magnitude col)
    idx is 1-based and follows CA residue order.
    """
    v = np.zeros((n_res, 3), dtype=float)
    with open(vec_path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                i = int(parts[0]) - 1
            except Exception:
                continue
            if 0 <= i < n_res:
                try:
                    v[i, 0] = float(parts[1])
                    v[i, 1] = float(parts[2])
                    v[i, 2] = float(parts[3])
                except Exception:
                    pass
    return v


# ------------------------- RigidParts TXT parsing -------------------------

def _split_res_token(tok: str) -> Tuple[Optional[int], str]:
    """
    tok may be: '57', '57A' (resnum + insertion code), etc.
    Returns: (resnum, icode_or_space)
    """
    tok = tok.strip()
    if not tok:
        return None, " "

    i = 0
    while i < len(tok) and tok[i].isdigit():
        i += 1
    if i == 0:
        return None, " "

    resnum = int(tok[:i])
    rest = tok[i:]
    icode = rest[:1] if rest else " "
    return resnum, (icode if icode else " ")


_RANGE_WITH_CHAIN = re.compile(r"([A-Za-z0-9])\s*:\s*([0-9]+[A-Za-z]?)\s*-\s*([0-9]+[A-Za-z]?)")
_RANGE_NO_CHAIN   = re.compile(r"\b([0-9]+[A-Za-z]?)\s*-\s*([0-9]+[A-Za-z]?)\b")


def _extract_ranges_from_text(s: str, default_chain: Optional[str]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns dict: chain -> [(startTok,endTok), ...] in order of appearance.
    Supports:
      - A:14-39,257-388
      - 14-39   (uses default_chain)
      - A:14-39  B:10-20  (rare, but supported)
    """
    out: Dict[str, List[Tuple[str, str]]] = {}

    found_any = False
    for m in _RANGE_WITH_CHAIN.finditer(s):
        found_any = True
        ch = m.group(1)
        a = m.group(2)
        b = m.group(3)
        out.setdefault(ch, []).append((a, b))

    if found_any:
        return out

    # no explicit chain ranges -> use default chain if given
    if default_chain:
        for m in _RANGE_NO_CHAIN.finditer(s):
            a = m.group(1)
            b = m.group(2)
            out.setdefault(default_chain, []).append((a, b))

    return out


def parse_rigidparts_txt(rigidparts_path: str) -> Dict[int, Dict[str, List[List[Tuple[str, str]]]]]:
    """
    Parse a rigidparts report (your output OR hingeprot-like output).

    Returns:
      parts_by_mode[mode][chain] = list_of_parts
      where each "part" is a list of ranges [(startTok,endTok), ...]
      Example:
        mode1, chainA:
          [
            [("1","39")],
            [("40","256")],
            [("257","409")],
            ...
          ]
      HingeProt style:
        Part 1 A:14-39,257-388 ...
        -> one part with two ranges: [("14","39"),("257","388")]
    """
    parts_by_mode: Dict[int, Dict[str, List[List[Tuple[str, str]]]]] = {}

    mode: Optional[int] = None
    cur_chain: Optional[str] = None
    in_table = False

    with open(rigidparts_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                in_table = False
                continue

            # mode header
            m = re.search(r"Slowest mode\s*([12])", line, flags=re.IGNORECASE)
            if line.startswith("---->") and m:
                mode = int(m.group(1))
                cur_chain = None
                in_table = False
                parts_by_mode.setdefault(mode, {})
                continue

            if mode is None:
                continue

            # chain line
            m = re.match(r"Chain\s+(\S+)", line, flags=re.IGNORECASE)
            if m:
                cur_chain = m.group(1).strip()[:1]
                continue

            # table header line
            if ("Rigid Part" in line and "Residues" in line) or ("Rigid Part No" in line and "Residues" in line):
                in_table = True
                continue

            # HingeProt-like "Part k ..." lines (not a table)
            if re.match(r"^Part\s+\d+\b", line, flags=re.IGNORECASE):
                rr = _extract_ranges_from_text(line, default_chain=cur_chain)
                for ch, ranges in rr.items():
                    parts_by_mode.setdefault(mode, {}).setdefault(ch, []).append(ranges)
                continue

            # Our table lines: "1\t\tA:1-39" or "1  A:1-39" or "1  1-39"
            if in_table:
                rr = _extract_ranges_from_text(line, default_chain=cur_chain)
                for ch, ranges in rr.items():
                    parts_by_mode.setdefault(mode, {}).setdefault(ch, []).append(ranges)
                continue

    # sanity: remove empty
    for md in list(parts_by_mode.keys()):
        for ch in list(parts_by_mode[md].keys()):
            if not parts_by_mode[md][ch]:
                del parts_by_mode[md][ch]
        if not parts_by_mode[md]:
            del parts_by_mode[md]

    if not parts_by_mode:
        raise ValueError(f"Could not parse any rigid parts from: {rigidparts_path}")

    return parts_by_mode


# ------------------------- B-factors from rigid parts -------------------------

def _build_chain_index_maps(ca_keys: List[ResKey]) -> Tuple[List[str], Dict[str, List[int]], Dict[str, List[ResKey]]]:
    chains = sorted(set(k.chain for k in ca_keys))
    global_indices_by_chain = {ch: [i for i, k in enumerate(ca_keys) if k.chain == ch] for ch in chains}
    chain_keys: Dict[str, List[ResKey]] = {ch: [ca_keys[i] for i in global_indices_by_chain[ch]] for ch in chains}
    return chains, global_indices_by_chain, chain_keys


def _find_pos_in_chain(keys: List[ResKey], res_tok: str, is_start: bool) -> Optional[int]:
    """
    Find best matching index for residue token in this chain.
    - Prefer exact match (resnum + insertion code if provided)
    - If not found: fallback to nearest by resnum (>= for start, <= for end)
    """
    resnum, icode = _split_res_token(res_tok)
    if resnum is None:
        return None

    ic = icode.strip()

    # exact
    if ic:
        for i, k in enumerate(keys):
            if k.resnum == resnum and k.icode.strip() == ic:
                return i
    else:
        for i, k in enumerate(keys):
            if k.resnum == resnum:
                return i

    # nearest fallback
    if is_start:
        for i, k in enumerate(keys):
            if k.resnum >= resnum:
                return i
        return len(keys) - 1 if keys else None
    else:
        for i in range(len(keys) - 1, -1, -1):
            if keys[i].resnum <= resnum:
                return i
        return 0 if keys else None


def build_bfactors_from_rigidparts_txt(
    ca_keys: List[ResKey],
    parts_by_mode: Dict[int, Dict[str, List[List[Tuple[str, str]]]]],
    bmag: float = 10.0,
) -> Dict[int, Dict[ResKey, float]]:
    """
    Assign B-factors using rigid parts read from txt.
    Rule:
      - Part1 => +bmag
      - Part2 => -bmag
      - Part3 => +bmag ...
    If a Part contains multiple disjoint ranges (hingeprot output), ALL ranges of that part share the same sign.
    """
    chains, global_indices_by_chain, chain_keys = _build_chain_index_maps(ca_keys)

    bfac_by_mode: Dict[int, Dict[ResKey, float]] = {1: {}, 2: {}}

    for mode in (1, 2):
        if mode not in parts_by_mode:
            continue

        for ch, parts in parts_by_mode[mode].items():
            if ch not in chain_keys:
                continue

            keys = chain_keys[ch]
            idxs_global = global_indices_by_chain[ch]

            for pi, ranges in enumerate(parts):
                sign = 1.0 if (pi % 2 == 0) else -1.0
                bval = float(bmag) * sign

                for (a_tok, b_tok) in ranges:
                    a = _find_pos_in_chain(keys, a_tok, is_start=True)
                    b = _find_pos_in_chain(keys, b_tok, is_start=False)
                    if a is None or b is None:
                        continue
                    if a > b:
                        a, b = b, a

                    for ci in range(a, b + 1):
                        g = idxs_global[ci]
                        bfac_by_mode[mode][ca_keys[g]] = bval

    return bfac_by_mode


# ------------------------- Writing moved PDB -------------------------

def write_moved_pdb(
    out_path: str,
    lines: List[str],
    atom_idx: List[int],
    atom_keys: List[ResKey],
    atom_xyz: np.ndarray,
    ca_keys: List[ResKey],
    vec: np.ndarray,
    bfac_map: Dict[ResKey, float],
    clustering_dist_thr: float,
):
    # scale displacement so step size matches previous behavior
    vec_mag = np.linalg.norm(vec, axis=1)
    max_mag = float(vec_mag.max()) if vec_mag.size else 1.0
    if max_mag <= 0.0:
        max_mag = 1.0
    per_step_target = float(clustering_dist_thr) / 5.0
    scale = per_step_target / max_mag

    resvec: Dict[ResKey, np.ndarray] = {k: vec[i] for i, k in enumerate(ca_keys)}

    def bfac_for_key(k: ResKey) -> float:
        return float(bfac_map.get(k, 0.0))

    factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    with open(out_path, "w", encoding="utf-8") as f:
        for mi, fac in enumerate(factors, start=1):
            f.write(f"MODEL{mi:9d}\n")
            out_lines = list(lines)

            for ai, li in enumerate(atom_idx):
                k = atom_keys[ai]
                disp = resvec.get(k)
                if disp is None:
                    continue

                dx, dy, dz = (disp * scale * float(fac))
                x0, y0, z0 = atom_xyz[ai]
                out_lines[li] = format_pdb_line_with_xyz_b(
                    out_lines[li],
                    x0 + dx, y0 + dy, z0 + dz,
                    bfac_for_key(k),
                )

            for line in out_lines:
                if line.startswith("MODEL") or line.startswith("ENDMDL"):
                    continue
                f.write(line.rstrip() + "\n")
            f.write("ENDMDL\n")


# ------------------------- CLI -------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("pdb_file")
    ap.add_argument("hinge_file", help="Put your generated rigidparts .txt here (e.g., 1v4s.rigidparts.txt)")
    ap.add_argument("vector_file1")
    ap.add_argument("vector_file2")
    ap.add_argument("loop_thr", type=int)  # backward-compatible (unused here)
    ap.add_argument("clustering_dist_thr", type=float)

    ap.add_argument("--bf_mag", type=float, default=10.0, help="Magnitude of B-factor values (+/-).")

    args = ap.parse_args(argv)

    lines, atom_idx, atom_keys, atom_xyz, ca_keys, _ca_xyz = read_pdb_atoms(args.pdb_file)

    # NEW: read rigid parts from txt and color based on those boundaries
    parts_by_mode = parse_rigidparts_txt(args.hinge_file)
    bfac_by_mode = build_bfactors_from_rigidparts_txt(
        ca_keys=ca_keys,
        parts_by_mode=parts_by_mode,
        bmag=args.bf_mag,
    )

    v1 = read_vector_file(args.vector_file1, n_res=len(ca_keys))
    v2 = read_vector_file(args.vector_file2, n_res=len(ca_keys))

    base, _ = os.path.splitext(args.pdb_file)
    out_m1 = base + ".moved1.pdb"
    out_m2 = base + ".moved2.pdb"

    write_moved_pdb(
        out_m1,
        lines, atom_idx, atom_keys, atom_xyz, ca_keys,
        v1,
        bfac_by_mode.get(1, {}),
        args.clustering_dist_thr,
    )
    write_moved_pdb(
        out_m2,
        lines, atom_idx, atom_keys, atom_xyz, ca_keys,
        v2,
        bfac_by_mode.get(2, {}),
        args.clustering_dist_thr,
    )

    print(f"Wrote: {out_m1}")
    print(f"Wrote: {out_m2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
