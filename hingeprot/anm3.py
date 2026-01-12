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
# Notes:
#   - The original FORTRAN later writes with format 31 but supplies fewer fields than the format expects.
#     Here we append three trailing 0.000 values to complete the line.
#   - The FORTRAN "SUEZ ADVA" block writes modes 17..42 into files 11coor..36coor, but it only read 36 modes.
#     This Python version preserves file creation and writes zeros for mode indices beyond what is present.

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
            # Parse by fixed columns (best effort) matching format 65
            # A6:  1-6
            # 1X:  7
            # I4:  8-11  -> slice [7:11]
            # 1X:  12
            # A4:  13-16 -> slice [12:16]
            # 1X:  17
            # A3:  18-20 -> slice [17:20]
            # 1X:  21
            # A1:  22     -> slice [21:22]
            # I4:  23-26  -> slice [22:26]
            # A1:  27     -> slice [26:27]
            # 3X:  28-30
            # 3F8.3: 31-38, 39-46, 47-54 -> slices [30:38], [38:46], [46:54]
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
        max_k = 1  # avoid division by zero later, mimic “something exists”

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
    w1: np.ndarray          # 1-based length (>=36+1)
    v1: np.ndarray          # shape (nmax+1, n_modes+1), 1-based indices
    n_modes: int


def parse_eigen_file(eig_path: str, max_modes: int = 36) -> EigenData:
    """
    Parse the unit=44 file.

    The FORTRAN logic:
      - skip 12 lines (A7)
      - read nmax from a line where format is (38x,I5)
      - find 'vector ' marker, then read eigenvalues until nn>=36
      - read 36 eigenvectors, each preceded by 'vector ' marker and one dummy line
      - eigenvector data comes in chunks of 6 floats (last line may be 3)
    """
    with open(eig_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    idx = 0

    # skip first 12 lines
    idx += 12
    if idx >= len(lines):
        raise RuntimeError(f"Eigen file {eig_path} is too short (missing header).")

    # read nmax from columns 39-43 (1-based) => slice [38:43]
    nmax_line = lines[idx].rstrip("\n")
    idx += 1
    nmax_str = _safe_slice(nmax_line, 38, 43).strip()
    if not nmax_str:
        # fallback: try to parse any integer in line
        toks = nmax_line.split()
        nmax_str = toks[-1] if toks else ""
    try:
        nmax = int(nmax_str)
    except ValueError:
        raise RuntimeError(f"Could not parse nmax from eigen file line: {nmax_line!r}")

    jres = nmax // 3

    # helper to find next 'vector ' line
    def find_next_vector(start: int) -> int:
        for i in range(start, len(lines)):
            s = lines[i].rstrip("\n")
            # FORTRAN reads A7 and compares to 'vector ' (7 chars)
            if _safe_slice(s, 0, 7) == "vector ":
                return i
        return -1

    # --- read eigenvalues w1 ---
    vec_i = find_next_vector(idx)
    if vec_i < 0:
        raise RuntimeError("Could not find 'vector ' marker for eigenvalues section.")
    idx = vec_i + 1  # next line
    if idx >= len(lines):
        raise RuntimeError("Unexpected EOF after eigenvalue 'vector ' marker.")
    idx += 1  # read(44,*) dummy  (skip one line)

    w1 = np.zeros(max_modes + 1, dtype=float)  # 1..max_modes
    # read nn, w1(nn), dumn until nn >= max_modes
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

    # --- read eigenvectors v1 ---
    # v1 is indexed v1(component_index, mode_index)
    # allocate for nmax components, up to max_modes modes
    v1 = np.zeros((nmax + 1, max_modes + 1), dtype=float)

    for mode_k in range(1, max_modes + 1):
        vec_i = find_next_vector(idx)
        if vec_i < 0:
            # fewer modes than expected; stop early
            n_modes = mode_k - 1
            break
        idx = vec_i + 1
        if idx >= len(lines):
            n_modes = mode_k - 1
            break
        idx += 1  # skip one dummy line after 'vector '

        filled = 0
        # fill indices 1..nmax
        while filled < nmax and idx < len(lines):
            parts = lines[idx].strip().split()
            idx += 1
            if not parts:
                continue
            # parse floats
            vals: List[float] = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
            if not vals:
                continue

            take = min(nmax - filled, len(vals))
            # FORTRAN stores to v1(n*6+j,k) with 1-based indexing
            start_comp = filled + 1
            end_comp = filled + take
            v1[start_comp:end_comp + 1, mode_k] = vals[:take]
            filled += take

        n_modes = mode_k  # last successfully read mode

    # If loop broke before setting n_modes:
    if "n_modes" not in locals():
        n_modes = max_modes

    return EigenData(nmax=nmax, jres=jres, w1=w1, v1=v1, n_modes=n_modes)


def write_eigenanm(outdir: str, eig: EigenData) -> None:
    path = os.path.join(outdir, "eigenanm")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(1, min(36, len(eig.w1) - 1) + 1):
            f.write(f"{i:4d}  {eig.w1[i]:8.4f}\n")


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

    # Open coor files
    coor_handles: Dict[int, Tuple[int, object]] = {}

    # 1coor..10coor correspond to modes k1..k10
    for idx_file, mode in enumerate([k1, k2, k3, k4, k5, k6, k7, k8, k9, k10], start=1):
        fh = open(os.path.join(outdir, f"{idx_file}coor"), "w", encoding="utf-8")
        coor_handles[idx_file] = (mode, fh)

    # 11coor..36coor: FORTRAN uses ss = k1 + ss_unit - 1101 where ss_unit=1111..1136 => ss=17..42
    suez_handles: Dict[int, Tuple[int, object]] = {}
    for file_no in range(11, 37):
        fh = open(os.path.join(outdir, f"{file_no}coor"), "w", encoding="utf-8")
        # Map file_no -> ss_unit: 1111 corresponds to file 11, ..., 1136 to file 36
        ss_unit = 1100 + file_no
        ss = k1 + ss_unit - 1101
        suez_handles[file_no] = (ss, fh)

    # Open cross files 1cross..10cross correspond to modes 7..16 (k=fileindex-67 for fileindex 74..83)
    cross_handles: Dict[int, Tuple[int, object]] = {}
    for cross_idx, mode in enumerate(range(7, 17), start=1):
        fh = open(os.path.join(outdir, f"{cross_idx}cross"), "w", encoding="utf-8")
        cross_handles[cross_idx] = (mode, fh)

    # Write coor files
    for j in range(1, jres + 1):
        ix = 3 * j - 2
        iy = 3 * j - 1
        iz = 3 * j

        # 1..10coor
        for file_no, (mode, fh) in coor_handles.items():
            if 1 <= mode <= n_modes and iz <= eig.nmax:
                x = float(v1[ix, mode])
                y = float(v1[iy, mode])
                z = float(v1[iz, mode])
            else:
                x = y = z = 0.0
            mag = math.sqrt(x * x + y * y + z * z)
            fh.write(fmt_9798(j, x, y, z, mag))

        # 11..36coor (SUEZ ADVA)
        for file_no, (mode, fh) in suez_handles.items():
            if 1 <= mode <= n_modes and iz <= eig.nmax:
                x = float(v1[ix, mode])
                y = float(v1[iy, mode])
                z = float(v1[iz, mode])
            else:
                x = y = z = 0.0
            mag = math.sqrt(x * x + y * y + z * z)
            fh.write(fmt_9798(j, x, y, z, mag))

    # Write cross files (cosine similarity between displacement vectors for each mode)
    for cross_idx, (mode, fh) in cross_handles.items():
        if not (1 <= mode <= n_modes):
            # still create file, but no content
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

    # Close all
    for _, (_, fh) in coor_handles.items():
        fh.close()
    for _, (_, fh) in suez_handles.items():
        fh.close()
    for _, (_, fh) in cross_handles.items():
        fh.close()


def write_newcoordinat(outdir: str, data: AlphaCorData, xxnew: List[float], yynew: List[float], zznew: List[float], jres: int) -> None:
    """
    Mimic FORTRAN write with format 31:
      31 format(a4,3x,I4,1x,a4,1x,a3,1x,a1,i4,a1,f11.3,f8.3,f8.3,
     :1x,f5.2,1x,i4,1x,f4.3,1x,f4.3,1x,f4.3)

    FORTRAN provided fewer args than required; we append 0.000 0.000 0.000.
    """
    path = os.path.join(outdir, "newcoordinat.mds")
    with open(path, "w", encoding="utf-8") as f:
        for j in range(1, jres + 1):
            # Safe access if j exceeds alpha arrays
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
                f" {0.0:4.3f}"
                f" {0.0:4.3f}"
                f" {0.0:4.3f}"
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

    # Read alpha.cor and compute centered coords
    alpha = parse_alpha_cor(args.alpha, max_size=5000)
    xxnew, yynew, zznew = compute_centered_coords(alpha)

    # Read eigen file (unit 44)
    eig = parse_eigen_file(args.eig, max_modes=36)

    # Write eigenanm
    write_eigenanm(outdir, eig)

    # Write 1coor..36coor and 1cross..10cross
    write_coor_and_cross(outdir, eig)

    # Re-read residue numbers into ind (as FORTRAN does)
    reread_residue_numbers(args.alpha, alpha)

    # Write newcoordinat.mds (jres comes from eigen file: nmax/3)
    write_newcoordinat(outdir, alpha, xxnew, yynew, zznew, eig.jres)


if __name__ == "__main__":
    main()
