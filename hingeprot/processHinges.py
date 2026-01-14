from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# ------------------------- PDB parsing / formatting -------------------------

@dataclass(frozen=True)
class ResKey:
    chain: str
    resnum: int
    icode: str  # insertion code (single char or space)


@dataclass(frozen=True)
class HingePoint:
    mode: int          # 1 or 2
    seq_idx: int       # 1-based residue-order index (from hinge file col-1). 0 => unknown
    chain: str
    resnum: int
    icode: str


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


# ------------------------- Hinge parsing (ROBUST) -------------------------

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


def parse_hinge_file(hinge_path: str) -> Dict[int, List[HingePoint]]:
    """
    Accepts BOTH formats inside a mode section:

    (A) 3 columns:
        <seq_idx> <resid> <chain>
        26   39   A

    (B) 2 columns:
        <resid> <chain>
        39   A

    seq_idx is used if present; otherwise seq_idx=0 and we later match by resnum/icode.
    """
    out: Dict[int, List[HingePoint]] = {1: [], 2: []}
    mode: Optional[int] = None

    with open(hinge_path, "r", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

            low = s.lower()
            if "1st slowest mode" in low:
                mode = 1
                continue
            if "2nd slowest mode" in low:
                mode = 2
                continue
            if s.startswith("---->"):
                continue
            if mode is None:
                continue

            parts = s.split()
            if len(parts) < 2:
                continue

            if len(parts) >= 3:
                # seq_idx resid chain
                try:
                    seq_idx = int(float(parts[0]))
                except Exception:
                    continue
                res_tok = parts[1]
                chain = (parts[2].strip()[:1] or "A")
            else:
                # resid chain
                seq_idx = 0
                res_tok = parts[0]
                chain = (parts[1].strip()[:1] or "A")

            resnum, icode = _split_res_token(res_tok)
            if resnum is None:
                continue

            out[mode].append(HingePoint(mode=mode, seq_idx=seq_idx, chain=chain, resnum=resnum, icode=icode))

    if not out[1] and not out[2]:
        raise ValueError("No hinge entries parsed from hinge file.")

    return out


def _res_label(k: ResKey) -> str:
    ic = (k.icode.strip() or "")
    return f"{k.resnum}{ic}{k.chain}"


# ------------------------- CORE: "kısa aralık varsa iki hinge de yok" -------------------------

def clean_hinges_ignore_short_fragments(
    chain_keys: List[ResKey],
    hinge_positions: List[int],
    min_seg_len: int,
) -> Tuple[List[int], List[Tuple[ResKey, ResKey]]]:
    """
    Senin istediğin mantık:

    - Eğer ardışık iki hinge arasında oluşacak segment uzunluğu < min_seg_len ise,
      o iki hinge de "yokmuş gibi" silinir. (işaret flip'i hiç yaşanmamış sayılır)
    - Baş segment (0..first_hinge) kısa ise -> first_hinge silinir
    - Kuyruk segment (last_hinge+1..end) kısa ise -> last_hinge silinir
    - Bu işlem, silmeler yeni komşuluklar doğurabileceği için iteratif yapılır.

    Returns:
      cleaned_positions: kalan hinge boundary indeksleri (0-based)
      short_pairs: hangi hinge-çiftleri (ve/veya baş/kuyruk) yüzünden silme olduğunu raporlamak için (left_res,right_res)
    """
    N = len(chain_keys)
    pos = sorted(set([p for p in hinge_positions if 0 <= p < N]))
    short_pairs_idx: List[Tuple[int, int]] = []

    if N == 0 or not pos:
        return pos, []

    while True:
        bad: set[int] = set()
        if not pos:
            break

        # start segment length = pos[0] + 1
        if (pos[0] + 1) < min_seg_len:
            bad.add(pos[0])
            short_pairs_idx.append((0, pos[0]))

        # between hinges segment length = pos[i] - pos[i-1]
        for i in range(1, len(pos)):
            if (pos[i] - pos[i - 1]) < min_seg_len:
                bad.add(pos[i - 1])
                bad.add(pos[i])
                short_pairs_idx.append((pos[i - 1], pos[i]))

        # tail segment length = (N-1) - pos[-1]
        if ((N - 1) - pos[-1]) < min_seg_len:
            bad.add(pos[-1])
            short_pairs_idx.append((pos[-1], N - 1))

        if not bad:
            break

        pos = [p for p in pos if p not in bad]

    # deduplicate + map to ResKey pairs
    uniq = []
    seen = set()
    for a, b in short_pairs_idx:
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            uniq.append(key)

    short_pairs: List[Tuple[ResKey, ResKey]] = [(chain_keys[a], chain_keys[b]) for a, b in uniq]
    return pos, short_pairs


def build_segments_from_hinges(N: int, hinges_pos: List[int]) -> List[Tuple[int, int]]:
    """
    hinge_pos: 0-based indices, each hinge is END of left segment.
    segments: [0..h1], [h1+1..h2], ..., [last+1..N-1]
    """
    if N <= 0:
        return []
    pos = sorted(set([p for p in hinges_pos if 0 <= p < N]))
    segs: List[Tuple[int, int]] = []
    start = 0
    for p in pos:
        if start <= p:
            segs.append((start, p))
        start = p + 1
    if start <= N - 1:
        segs.append((start, N - 1))
    if not segs:
        segs = [(0, N - 1)]
    return segs


# ------------------------- B-factors from FINAL rigid parts -------------------------

def build_bfactors_from_hinges(
    ca_keys: List[ResKey],
    hinges: Dict[int, List[HingePoint]],
    min_seg_len: int = 15,
    bmag: float = 10.0,
) -> Tuple[
    Dict[int, Dict[ResKey, float]],
    Dict[int, Dict[str, List[Tuple[int, int]]]],
    Dict[int, Dict[str, List[str]]]
]:
    """
    Returns:
      bfac_by_mode: {mode -> {ResKey -> bfac}}
      report_by_mode:
         report_by_mode[mode][chain] = [(start_resnum,end_resnum), ...] rigid parts
         report_by_mode[mode][f"{chain}__SFF"] = [(left_resnum,right_resnum), ...] short-cluster pairs (diagnostic)
      hinge_report:
         hinge_report[mode][f"{chain}__KEPT"] = [labels...]
         hinge_report[mode][f"{chain}__IGNORED"] = [labels...]
    """
    chains = sorted(set(k.chain for k in ca_keys))
    chain_keys: Dict[str, List[ResKey]] = {ch: [k for k in ca_keys if k.chain == ch] for ch in chains}

    # global indices per chain (to map chain-local positions -> global CA index)
    global_indices_by_chain = {ch: [i for i, k in enumerate(ca_keys) if k.chain == ch] for ch in chains}
    chain_pos_of_global: Dict[str, Dict[int, int]] = {ch: {} for ch in chains}
    for ch in chains:
        idxs = global_indices_by_chain[ch]
        chain_pos_of_global[ch] = {g: ci for ci, g in enumerate(idxs)}

    bfac_by_mode: Dict[int, Dict[ResKey, float]] = {1: {}, 2: {}}
    report_by_mode: Dict[int, Dict[str, List[Tuple[int, int]]]] = {1: {}, 2: {}}
    hinge_report: Dict[int, Dict[str, List[str]]] = {1: {}, 2: {}}

    for mode in (1, 2):
        hinges_mode = sorted(hinges.get(mode, []), key=lambda h: (h.seq_idx if h.seq_idx > 0 else 10**9, h.resnum))

        for ch in chains:
            keys = chain_keys[ch]
            N = len(keys)
            if N == 0:
                continue

            # candidate hinge positions in chain coords
            cand_chainpos: List[int] = []

            for hp in hinges_mode:
                if hp.chain != ch:
                    continue

                # 1) Prefer seq_idx mapping if available and consistent
                if hp.seq_idx > 0:
                    gpos = hp.seq_idx - 1
                    if 0 <= gpos < len(ca_keys) and ca_keys[gpos].chain == ch:
                        cand_chainpos.append(chain_pos_of_global[ch][gpos])
                        continue

                # 2) Fallback: match by resnum/icode in this chain
                for ci, kk in enumerate(keys):
                    if kk.resnum == hp.resnum:
                        if (hp.icode.strip() == "") or (kk.icode.strip() == hp.icode.strip()):
                            cand_chainpos.append(ci)
                            break

            cand_chainpos = sorted(set([p for p in cand_chainpos if 0 <= p < N]))

            # === FIXED HERE: short pair => BOTH hinges are ignored ===
            cleaned_pos, short_pairs = clean_hinges_ignore_short_fragments(
                chain_keys=keys,
                hinge_positions=cand_chainpos,
                min_seg_len=min_seg_len,
            )

            segs = build_segments_from_hinges(N, cleaned_pos)

            # report segments
            report_by_mode[mode][ch] = [(keys[a].resnum, keys[b].resnum) for a, b in segs]
            if short_pairs:
                report_by_mode[mode][f"{ch}__SFF"] = [(a.resnum, b.resnum) for a, b in short_pairs]

            kept_set = set(cleaned_pos)
            all_set = set(cand_chainpos)
            ignored_set = sorted(list(all_set - kept_set))

            hinge_report[mode][f"{ch}__KEPT"] = [_res_label(keys[p]) for p in sorted(list(kept_set))]
            hinge_report[mode][f"{ch}__IGNORED"] = [_res_label(keys[p]) for p in ignored_set]

            # assign B-factors by FINAL rigid parts (flip only at cleaned_pos boundaries)
            idxs_global = global_indices_by_chain[ch]
            for si, (a, b) in enumerate(segs):
                sign = 1.0 if (si % 2 == 0) else -1.0
                bval = float(bmag) * sign
                for ci in range(a, b + 1):
                    g = idxs_global[ci]
                    bfac_by_mode[mode][ca_keys[g]] = bval

    return bfac_by_mode, report_by_mode, hinge_report


def write_rigidparts_report(
    out_path: str,
    report_by_mode: Dict[int, Dict[str, List[Tuple[int, int]]]],
    hinge_report: Dict[int, Dict[str, List[str]]],
    ca_keys: List[ResKey],
    hinges: Dict[int, List[HingePoint]],
):
    with open(out_path, "w", encoding="utf-8") as f:
        for mode in (1, 2):
            f.write(f"----> Slowest mode {mode}:\n")

            for ch in sorted(set(k.chain for k in ca_keys)):
                segs = report_by_mode.get(mode, {}).get(ch, [])
                f.write(f"Chain {ch}\n")
                f.write("Rigid Part No\tResidues\n")
                for i, (a, b) in enumerate(segs, start=1):
                    f.write(f"{i}\t\t{ch}:{a}-{b}\n")

                kept = hinge_report.get(mode, {}).get(f"{ch}__KEPT", [])
                ign = hinge_report.get(mode, {}).get(f"{ch}__IGNORED", [])

                f.write("Hinge residues (KEPT boundaries): " + (" ".join(kept) if kept else "(none)") + "\n")
                if ign:
                    f.write("Hinge residues (IGNORED; treated as if they never existed): " + " ".join(ign) + "\n")

                sff = report_by_mode.get(mode, {}).get(f"{ch}__SFF", [])
                if sff:
                    f.write("Short-cluster pairs that caused IGNORE (diagnostic):\n")
                    for (a, b) in sff:
                        f.write(f"{ch}:{a}-{b}\n")

                f.write("\n")

            f.write("\n")


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
    ap.add_argument("hinge_file")
    ap.add_argument("vector_file1")
    ap.add_argument("vector_file2")
    ap.add_argument("loop_thr", type=int)  # backward-compatible (unused here)
    ap.add_argument("clustering_dist_thr", type=float)

    ap.add_argument("--min_seg_len", type=int, default=15,
                    help="If any segment induced by consecutive hinges is shorter than this, BOTH hinges are ignored.")
    ap.add_argument("--bf_mag", type=float, default=10.0, help="Magnitude of B-factor values (+/-).")
    ap.add_argument("--write_rigidparts", action="store_true", help="Write <pdb>.rigidparts.txt report.")

    args = ap.parse_args(argv)

    lines, atom_idx, atom_keys, atom_xyz, ca_keys, _ca_xyz = read_pdb_atoms(args.pdb_file)
    hinges = parse_hinge_file(args.hinge_file)

    bfac_by_mode, report_by_mode, hinge_report = build_bfactors_from_hinges(
        ca_keys,
        hinges,
        min_seg_len=args.min_seg_len,
        bmag=args.bf_mag,
    )

    v1 = read_vector_file(args.vector_file1, n_res=len(ca_keys))
    v2 = read_vector_file(args.vector_file2, n_res=len(ca_keys))

    base, _ = os.path.splitext(args.pdb_file)
    out_m1 = base + ".moved1.pdb"
    out_m2 = base + ".moved2.pdb"
    out_rp = base + ".rigidparts.txt"

    write_moved_pdb(
        out_m1,
        lines, atom_idx, atom_keys, atom_xyz, ca_keys,
        v1,
        bfac_by_mode[1],
        args.clustering_dist_thr,
    )
    write_moved_pdb(
        out_m2,
        lines, atom_idx, atom_keys, atom_xyz, ca_keys,
        v2,
        bfac_by_mode[2],
        args.clustering_dist_thr,
    )

    if args.write_rigidparts:
        write_rigidparts_report(out_rp, report_by_mode, hinge_report, ca_keys, hinges)
        print(f"Wrote: {out_rp}")

    print(f"Wrote: {out_m1}")
    print(f"Wrote: {out_m2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
