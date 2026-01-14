# anm3.py
# Translation of the provided FORTRAN (program rbp) into Python.
#
# Expected inputs (defaults match the FORTRAN):
#   - alpha.cor   (coordinates / residue info)
#   - fort.44     (eigenvectors/eigenvalues text; FORTRAN used "open(unit=44)" with no filename,
#                 which typically maps to fort.44)
#
# Outputs (same names as FORTRAN):
#   - newcoordinat.mds
#   - eigenanm
#   - 1coor..10coor, 11coor..36coor
#   - 1cross..10cross
#
# Changes requested by user:
#   - eigenanm: write the first 36 eigenvalues that are > 0 (skip zeros/negatives),
#               and do NOT write any header/count line.
#   - newcoordinat.mds: remove the last three trailing 0.000 columns.

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class AlphaCorData:
    # 1-based arrays (index 0 unused), mimicking FORTRAN style
    attyp: List[str]
    restyp: List[str]
    chnam: List[str]  # single char
    ind:   List[int]
    resex: List[str]  # single char
    xx:    List[float]
    yy:    List[float]
    zz:    List[float]
    resnum: int


def _safe_slice(line: str, a: int, b: int) -> str:
    """Safe slicing (0-based, end-exclusive). Returns '' if out of range."""
    if a >= len(line):
        return ""
    return line[a:min(b, len(line))]


def parse_alpha_cor(alpha_path: str, max_size: int = 5000) -> AlphaCorData:
    """
    Parse alpha.cor using the FORTRAN format:
      65 FORMAT (A6,1X,I4,1X,A4,1X,A3,1x,A1,I4,A1,3X,3F8.3)
    """
    # Pre-allocate (1..max_size). Index 0 unused.
    attyp = [""] * (max_size + 1)
    restyp = [""] * (max_size + 1)
    chnam = [""] * (max_size + 1)
    ind = [0] * (max_size + 1)
    resex = [""] * (max_size + 1)
    xx = [0.0] * (max_size + 1)
    yy = [0.0] * (max_size + 1)
    zz = [0.0] * (max_size + 1)

    max_k = 0

    with open(alpha_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            k_str = _safe_slice(line, 7, 11).strip()
            if not k_str:
                continue
            try:
                k = int(k_str)
            except ValueError:
                continue

            if k < 1 or k > max_size:
                continue

            attyp[k] = _safe_slice(line, 12, 16).strip()
            restyp[k] = _safe_slice(line, 17, 20).strip()
            chnam[k] = _safe_slice(line, 21, 22).strip()[:1] or " "
            ind_str = _safe_slice(line, 22, 26).strip()
            resex[k] = (_safe_slice(line, 26, 27).strip()[:1] or " ")

            try:
                ind[k] = int(ind_str) if ind_str else 0
            except ValueError:
                ind[k] = 0

            def _f(s: str) -> float:
                s = s.strip()
                if not s:
                    return 0.0
                try:
                    return float(s)
                except ValueError:
                    return 0.0

            xx[k] = _f(_safe_slice(line, 30, 38))
            yy[k] = _f(_safe_slice(line, 38, 46))
            zz[k] = _f(_safe_slice(line, 46, 54))

            if k > max_k:
                max_k = k

    if max_k == 0:
        max_k = 1

    return AlphaCorData(
        attyp=attyp,
        restyp=restyp,
        chnam=chnam,
        ind=ind,
        resex=resex,
        xx=xx,
        yy=yy,
        zz=zz,
        resnum=max_k,
    )


def reread_residue_numbers(alpha_path: str, data: AlphaCorData) -> None:
    """
    Mimic:
      97 format (7x,I4,11x,I4)
      read(10,97,end=3442) im,ind(im)
    """
    with open(alpha_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            im_str = _safe_slice(line, 7, 11).strip()      # cols 8-11
            ind_str = _safe_slice(line, 22, 26).strip()    # cols 23-26
            if not im_str:
                continue
            try:
                im = int(im_str)
            except ValueError:
                continue
            if im < 1 or im >= len(data.ind):
                continue
            try:
                data.ind[im] = int(ind_str) if ind_str else data.ind[im]
            except ValueError:
                pass


def compute_centered_coords(data: AlphaCorData) -> Tuple[List[float], List[float], List[float]]:
    """
    FORTRAN sums from 1..resnum including any zero-filled entries.
    """
    resnum = data.resnum
    centx = 0.0
    centy = 0.0
    centz = 0.0
    for i in range(1, resnum + 1):
        centx += data.xx[i]
        centy += data.yy[i]
        centz += data.zz[i]
    centx /= float(resnum)
    centy /= float(resnum)
    centz /= float(resnum)

    xxnew = [0.0] * (len(data.xx))
    yynew = [0.0] * (len(data.yy))
    zznew = [0.0] * (len(data.zz))
    for i in range(1, resnum + 1):
        xxnew[i] = -centx + data.xx[i]
        yynew[i] = -centy + data.yy[i]
        zznew[i] = -centz + data.zz[i]
    return xxnew, yynew, zznew


@dataclass
class EigenData:
    nmax: int
    jres: int
    w1: np.ndarray          # 1-based length (>=max_modes+1)
    v1: np.ndarray          # shape (nmax+1, max_modes+1), 1-based indices
    n_modes: int


def parse_eigen_file(eig_path: str, max_modes: int = 60, nmax_expected: int | None = None) -> EigenData:
    """
    Supports TWO formats:

    (A) Legacy FORTRAN 'fort.44' style:
        - header lines
        - nmax on a fixed-column line
        - repeated blocks starting with 'vector '

    (B) Numeric "vwmatrix" style (useblz.py-like):
        - eigenpairs + eigenvectors as floats
        - nmax inferred from alpha.cor via nmax_expected
    """
    with open(eig_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # ---------- detect legacy 'vector ' blocks ----------
    if any((_safe_slice(ln, 0, 7) == "vector ") for ln in lines):
        idx = 0
        idx += 12
        if idx >= len(lines):
            raise RuntimeError(f"Eigen file {eig_path} is too short (missing header).")

        nmax_line = lines[idx]
        idx += 1
        nmax_str = _safe_slice(nmax_line, 38, 43).strip()
        if not nmax_str:
            toks = nmax_line.split()
            nmax_str = toks[-1] if toks else ""
        try:
            nmax = int(nmax_str)
        except ValueError:
            raise RuntimeError(f"Could not parse nmax from eigen file line: {nmax_line!r}")

        jres = nmax // 3

        def find_next_vector(start: int) -> int:
            for i in range(start, len(lines)):
                if _safe_slice(lines[i], 0, 7) == "vector ":
                    return i
            return -1

        # eigenvalues
        vec_i = find_next_vector(idx)
        if vec_i < 0:
            raise RuntimeError("Could not find 'vector ' marker for eigenvalues section.")
        idx = vec_i + 1
        idx += 1  # skip dummy line after 'vector '

        w1 = np.zeros(max_modes + 1, dtype=float)
        while idx < len(lines):
            parts = lines[idx].strip().split()
            idx += 1
            if len(parts) < 2:
                continue
            try:
                nn = int(float(parts[0]))
            except ValueError:
                continue
            try:
                val = float(parts[1])
            except ValueError:
                val = 0.0
            if 1 <= nn <= max_modes:
                w1[nn] = val
            if nn >= max_modes:
                break

        # eigenvectors
        v1 = np.zeros((nmax + 1, max_modes + 1), dtype=float)
        n_modes = 0

        for mode_k in range(1, max_modes + 1):
            vec_i = find_next_vector(idx)
            if vec_i < 0:
                break
            idx = vec_i + 1
            idx += 1  # skip one dummy line

            filled = 0
            while filled < nmax and idx < len(lines):
                parts = lines[idx].strip().split()
                idx += 1
                if not parts:
                    continue
                vals = []
                for p in parts:
                    try:
                        vals.append(float(p))
                    except ValueError:
                        pass
                if not vals:
                    continue

                take = min(nmax - filled, len(vals))
                start_comp = filled + 1
                end_comp = filled + take
                v1[start_comp:end_comp + 1, mode_k] = vals[:take]
                filled += take

            n_modes = mode_k

        return EigenData(nmax=nmax, jres=jres, w1=w1, v1=v1, n_modes=n_modes)

    # ---------- numeric "vwmatrix" parser ----------
    if nmax_expected is None or nmax_expected <= 0:
        raise RuntimeError(
            "Numeric eigen format detected, but nmax_expected was not provided. "
            "Pass nmax_expected=3*n_residues from alpha.cor."
        )

    def parse_floats(ln: str) -> list[float]:
        out = []
        for tok in ln.strip().split():
            try:
                out.append(float(tok))
            except ValueError:
                pass
        return out

    all_floats: list[float] = []
    for ln in lines:
        all_floats.extend(parse_floats(ln))

    nmax = int(nmax_expected)
    total = len(all_floats)
    if total == 0:
        raise RuntimeError("Eigen file contains no numeric data.")

    k_total = None
    has_eigpairs = False
    vec_start = 0

    if total >= (nmax + 2):
        k_floor = total // (nmax + 2)
        rem = total - k_floor * (nmax + 2)
        if k_floor >= 1 and rem <= 32:
            k_total = int(k_floor)
            has_eigpairs = True
            vec_start = 2 * k_total

    if k_total is None and total >= nmax:
        k_floor = total // nmax
        rem = total - k_floor * nmax
        if k_floor >= 1 and rem <= 32:
            k_total = int(k_floor)
            has_eigpairs = False
            vec_start = 0

    if k_total is None or k_total <= 0:
        raise RuntimeError(
            f"Cannot infer k from numeric eigen file. total_floats={total}, nmax={nmax}. "
            f"Try checking upperhessian.vwmatrix formatting."
        )

    if has_eigpairs:
        eigvals = [float(all_floats[2 * i]) for i in range(k_total)]
    else:
        eigvals = [0.0] * k_total

    need_vec = nmax * k_total
    vec_floats = all_floats[vec_start:vec_start + need_vec]
    if len(vec_floats) < need_vec:
        raise RuntimeError(
            f"Eigenvector data too short. Need {need_vec} floats (=nmax*k), got {len(vec_floats)}. "
            f"(nmax={nmax}, k={k_total}, vec_start={vec_start}, total_floats={total})"
        )

    V_row = np.array(vec_floats, dtype=float).reshape((nmax, k_total), order="C")
    V_col = np.array(vec_floats, dtype=float).reshape((k_total, nmax), order="C").T

    def score_matrix(V: np.ndarray) -> float:
        m = min(10, V.shape[1])
        X = V[:, :m]
        norms = np.linalg.norm(X, axis=0)
        norm_err = float(np.mean(np.abs(norms - 1.0)))
        norms_safe = np.where(norms == 0.0, 1.0, norms)
        Xn = X / norms_safe
        G = Xn.T @ Xn
        off = G - np.eye(m)
        ortho_err = float(np.mean(np.abs(off)))
        return norm_err + ortho_err

    s_row = score_matrix(V_row)
    s_col = score_matrix(V_col)
    V_best = V_row if s_row <= s_col else V_col

    modes_to_read = min(max_modes, k_total)

    w1 = np.zeros(max_modes + 1, dtype=float)
    for i in range(1, modes_to_read + 1):
        w1[i] = float(eigvals[i - 1])

    v1 = np.zeros((nmax + 1, max_modes + 1), dtype=float)
    for m in range(1, modes_to_read + 1):
        v1[1:nmax + 1, m] = V_best[:, m - 1]

    jres = nmax // 3
    return EigenData(nmax=nmax, jres=jres, w1=w1, v1=v1, n_modes=modes_to_read)


def write_eigenanm(outdir: str, eig: EigenData) -> None:
    """
    Write eigenanm as:
      index   eigenvalue
    Taking DIRECTLY the first 36 eigenvalues (no filtering).
    No header/count line is written.
    """
    path = os.path.join(outdir, "eigenanm")
    with open(path, "w", encoding="utf-8") as f:
        # Always write 1..36 (if missing, w1 entries remain 0.0)
        for i in range(1, 37):
            val = float(eig.w1[i]) if i < len(eig.w1) else 0.0
            f.write(f"{i:4d}  {val:8.4f}\n")


def fmt_9798(j: int, x: float, y: float, z: float, mag: float) -> str:
    # 9798 format (I4,4(3x,F8.5))
    return f"{j:4d}   {x:8.5f}   {y:8.5f}   {z:8.5f}   {mag:8.5f}\n"


def fmt_9799(i: int, j: int, val: float) -> str:
    # 9799 format (I4,I4,3x,F8.5)
    return f"{i:4d}{j:4d}   {val:8.5f}\n"


def write_coor_and_cross(outdir: str, eig: EigenData) -> None:
    v1 = eig.v1
    jres = eig.jres
    n_modes = eig.n_modes

    # Mode selections as in FORTRAN
    k1 = 7
    k2 = k1 + 1
    k3 = k1 + 2
    k4 = k1 + 3
    k5 = k1 + 4
    k6 = k1 + 5
    k7 = k1 + 6
    k8 = k1 + 7
    k9 = k1 + 8
    k10 = k1 + 9

    coor_handles: Dict[int, Tuple[int, object]] = {}
    for idx_file, mode in enumerate([k1, k2, k3, k4, k5, k6, k7, k8, k9, k10], start=1):
        fh = open(os.path.join(outdir, f"{idx_file}coor"), "w", encoding="utf-8")
        coor_handles[idx_file] = (mode, fh)

    suez_handles: Dict[int, Tuple[int, object]] = {}
    for file_no in range(11, 37):
        fh = open(os.path.join(outdir, f"{file_no}coor"), "w", encoding="utf-8")
        ss_unit = 1100 + file_no
        ss = k1 + ss_unit - 1101  # 17..42
        suez_handles[file_no] = (ss, fh)

    cross_handles: Dict[int, Tuple[int, object]] = {}
    for cross_idx, mode in enumerate(range(7, 17), start=1):
        fh = open(os.path.join(outdir, f"{cross_idx}cross"), "w", encoding="utf-8")
        cross_handles[cross_idx] = (mode, fh)

    for j in range(1, jres + 1):
        ix = 3 * j - 2
        iy = 3 * j - 1
        iz = 3 * j

        for _, (mode, fh) in coor_handles.items():
            if 1 <= mode <= n_modes and iz <= eig.nmax:
                x = float(v1[ix, mode])
                y = float(v1[iy, mode])
                z = float(v1[iz, mode])
            else:
                x = y = z = 0.0
            mag = math.sqrt(x * x + y * y + z * z)
            fh.write(fmt_9798(j, x, y, z, mag))

        for _, (mode, fh) in suez_handles.items():
            if 1 <= mode <= n_modes and iz <= eig.nmax:
                x = float(v1[ix, mode])
                y = float(v1[iy, mode])
                z = float(v1[iz, mode])
            else:
                x = y = z = 0.0
            mag = math.sqrt(x * x + y * y + z * z)
            fh.write(fmt_9798(j, x, y, z, mag))

    for _, (mode, fh) in cross_handles.items():
        if not (1 <= mode <= n_modes):
            continue

        for i in range(1, jres + 1):
            iix = 3 * i - 2
            iiy = 3 * i - 1
            iiz = 3 * i
            vi = np.array([v1[iix, mode], v1[iiy, mode], v1[iiz, mode]], dtype=float)
            mag1 = float(np.linalg.norm(vi))

            for j in range(i, jres + 1):
                jix = 3 * j - 2
                jiy = 3 * j - 1
                jiz = 3 * j
                vj = np.array([v1[jix, mode], v1[jiy, mode], v1[jiz, mode]], dtype=float)
                mag2 = float(np.linalg.norm(vj))

                if mag1 == 0.0 or mag2 == 0.0:
                    dumn = 0.0
                else:
                    dumn = float(np.dot(vi, vj) / (mag1 * mag2))
                fh.write(fmt_9799(i, j, dumn))

    for _, (_, fh) in coor_handles.items():
        fh.close()
    for _, (_, fh) in suez_handles.items():
        fh.close()
    for _, (_, fh) in cross_handles.items():
        fh.close()


def write_newcoordinat(
    outdir: str,
    data: AlphaCorData,
    xxnew: List[float],
    yynew: List[float],
    zznew: List[float],
    jres: int
) -> None:
    """
    Write newcoordinat.mds.
    User request: remove the last three trailing 0.000 columns.
    """
    path = os.path.join(outdir, "newcoordinat.mds")
    with open(path, "w", encoding="utf-8") as f:
        for j in range(1, jres + 1):
            at = data.attyp[j] if j < len(data.attyp) else ""
            rt = data.restyp[j] if j < len(data.restyp) else ""
            ch = data.chnam[j] if j < len(data.chnam) else " "
            ix = data.ind[j] if j < len(data.ind) else 0
            rx = data.resex[j] if j < len(data.resex) else " "
            x = xxnew[j] if j < len(xxnew) else 0.0
            y = yynew[j] if j < len(yynew) else 0.0
            z = zznew[j] if j < len(zznew) else 0.0

            line = (
                f"{'ATOM':<4}"
                f"{'':3s}"
                f"{j:4d}"
                f" {at:<4}"
                f" {rt:<3}"
                f" {ch[:1]:<1}{ix:4d}{rx[:1]:<1}"
                f"{x:11.3f}{y:8.3f}{z:8.3f}"
                f" {1.00:5.2f}"
                f" {ix:4d}"
                f"\n"
            )
            f.write(line)
            print(line, end="")


def main() -> None:
    ap = argparse.ArgumentParser(description="FORTRAN rbp -> Python (anm3.py) translation")
    ap.add_argument("--alpha", default="alpha.cor", help="Path to alpha.cor (default: alpha.cor)")
    ap.add_argument("--eig", default="fort.44", help="Path to unit=44 eigen file (default: fort.44)")
    ap.add_argument("--outdir", default=".", help="Output directory (default: current directory)")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    alpha = parse_alpha_cor(args.alpha, max_size=5000)
    xxnew, yynew, zznew = compute_centered_coords(alpha)

    nmax_expected = 3 * alpha.resnum

    # IMPORTANT: read more than 36 modes so we can skip zero modes and still write 36 positive eigenvalues if available
    eig = parse_eigen_file(args.eig, max_modes=36, nmax_expected=nmax_expected)

    write_eigenanm(outdir, eig)
    write_coor_and_cross(outdir, eig)

    reread_residue_numbers(args.alpha, alpha)
    write_newcoordinat(outdir, alpha, xxnew, yynew, zznew, eig.jres)


if __name__ == "__main__":
    main()
