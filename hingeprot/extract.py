#!/usr/bin/env python3
"""
extract.py  (converted from extract.f / program hingebend)

Goal: match the provided Fortran behavior as closely as possible.

Expects files (same as Fortran):
  rescale
  coordinates
  alpha.cor
  1coor .. 10coor
  newcoordinat.mds
  slowmodes
  crosscorrslow1
  crosscorrslow2

Produces:
  anm_length
  newcoordinat2.mds
  mapping.out
  coor1.mds12 coor2.mds12 coor3.mds12 coor4.mds12
  gnm1anmvector gnm2anmvector
  hinges
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# ---------------------------- helpers ----------------------------

def _as_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _as_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except Exception:
        return default


def _read_first_float(path: str | Path) -> float:
    txt = Path(path).read_text(encoding="utf-8", errors="replace").strip().split()
    if not txt:
        raise RuntimeError(f"Empty file: {path}")
    return float(txt[0])


def _read_first_int(path: str | Path) -> int:
    txt = Path(path).read_text(encoding="utf-8", errors="replace").strip().split()
    if not txt:
        raise RuntimeError(f"Empty file: {path}")
    return int(float(txt[0]))


# ---------------------------- format parsers --------------------------------
# Fortran formats:
# 97  format (17x,A3,2x,I4)     -> alpha.cor: residue type + cano
# 98  format (4x,4(3x,F8.5))    -> coor files: (skip 4 chars = index), then x y z boy
# 99  format (I4,2x,I4,10(3x,F8.5))
# 100 format (a4,8x,a4,1x,a3,1x,a1,i4,a1,f11.3,f8.3,f8.3,1x,f5.2)  (Fortran may ignore occ on read/write)
# 101 format (I4,4(3x,F8.5))
# 16  format(4x,10(4x,F8.5))    -> slowmodes rows


@dataclass
class AtomRow:
    label: str      # a4
    attyp: str      # a4
    restyp: str     # a3
    chtyp: str      # a1 (Fortran variable was char*2 but format reads 1 char)
    ind: int        # i4
    resex: str      # a1
    x: float        # f11.3
    y: float        # f8.3
    z: float        # f8.3
    occ: float = 0.0  # optional f5.2


def parse_alpha_line(line: str) -> Tuple[str, int]:
    # 97: (17x,A3,2x,I4)
    restyp = line[17:20].strip()
    cano = _as_int(line[22:26].strip(), 0)
    return restyp, cano


def parse_coor_line(line: str) -> Tuple[float, float, float, float]:
    """
    Fortran 98 reads: 4x then 4 floats.
    The file line is typically written by rbp/anm stage as:
      I4 + 4 floats (x y z boy)  -> split gives 5 tokens
    Fortran skips the first 4 chars (the index field). So we must ignore the first token if present.
    """
    parts = line.strip().split()
    if len(parts) >= 5:
        # common case: [index, x, y, z, boy]
        # verify first token is integer-ish; if not, still treat as index to mimic 4x skip
        return (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    if len(parts) == 4:
        # rare case: no index token
        return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))

    # fixed-slice fallback (best effort)
    # If line has index in cols 1-4, floats likely start after that; attempt from known widths
    # This is only used if split fails.
    x = _as_float(line[7:15].strip(), 0.0)
    y = _as_float(line[18:26].strip(), 0.0)
    z = _as_float(line[29:37].strip(), 0.0)
    b = _as_float(line[40:48].strip(), 0.0)
    return (x, y, z, b)


def parse_mds100(line: str) -> AtomRow:
    """
    Parse using fixed columns for format 100.
    Note: Fortran sometimes ignores occ (f5.2) if not in I/O list,
    but the record may still contain it. We parse it if present; otherwise 0.0.
    """
    # make line long enough to slice safely
    s = line.rstrip("\n")
    s_pad = s + (" " * max(0, 80 - len(s)))

    label = s_pad[0:4].strip()
    attyp = s_pad[12:16].strip()
    restyp = s_pad[17:20].strip()
    chtyp = s_pad[21:22].strip()
    ind = _as_int(s_pad[22:26].strip(), 0)
    resex = s_pad[26:27].strip()
    x = _as_float(s_pad[27:38].strip(), 0.0)
    y = _as_float(s_pad[38:46].strip(), 0.0)
    z = _as_float(s_pad[46:54].strip(), 0.0)
    occ = _as_float(s_pad[55:60].strip(), 0.0)  # may be absent/blank
    return AtomRow(label, attyp, restyp, chtyp, ind, resex, x, y, z, occ)


def format_mds100(row: AtomRow, include_occ: bool) -> str:
    """
    Mimic Fortran format 100:
      a4,8x,a4,1x,a3,1x,a1,i4,a1,f11.3,f8.3,f8.3,(optional) 1x,f5.2
    In Fortran, if occ is not in the I/O list, the record ends after z.
    So we support include_occ=False to match write(35,100) behavior.
    """
    base = (
        f"{row.label:<4}"
        + " " * 8
        + f"{row.attyp:<4} "
        + f"{row.restyp:<3} "
        + f"{row.chtyp:<1}"
        + f"{row.ind:4d}"
        + f"{row.resex:<1}"
        + f"{row.x:11.3f}{row.y:8.3f}{row.z:8.3f}"
    )
    if include_occ:
        base += f" {row.occ:5.2f}"
    return base + "\n"


def format_101(i: int, x: float, y: float, z: float, boy: float) -> str:
    # 101: (I4,4(3x,F8.5))
    return (
        f"{i:4d}"
        f"   {x:8.5f}"
        f"   {y:8.5f}"
        f"   {z:8.5f}"
        f"   {boy:8.5f}\n"
    )


def format_99(i: int, cano: int, boys: List[float]) -> str:
    # 99: (I4,2x,I4,10(3x,F8.5))
    s = f"{i:4d}  {cano:4d}"
    for v in boys:
        s += f"   {v:8.5f}"
    return s + "\n"


def read_slowmodes(path: str | Path, resno: int) -> List[List[float]]:
    # returns gnmboy[j][i] with j=0..9, i=0..resno-1
    gnmboy = [[0.0 for _ in range(resno)] for _ in range(10)]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i in range(resno):
            line = f.readline()
            if not line:
                raise RuntimeError(f"slowmodes ended early at residue {i+1}/{resno}")
            vals = [float(x) for x in line.strip().split()]
            if len(vals) < 10:
                vals = vals + [0.0] * (10 - len(vals))
            for j in range(10):
                gnmboy[j][i] = vals[j]
    return gnmboy


# ---------------------------- hinge finder -----------------------------------

def findhinge(infile: str | Path, resno: int) -> List[int]:
    """
    Mirror Fortran subroutine findhinge EXACTLY:
    - read first 13 chars, check (11:13) == 'nan'
    - backspace (i.e., re-process first line)
    - read formatted triples i j value sequentially
    - STOP as soon as j == resno for the first time
    - hinges where sign changes between mat(j) and mat(j+1)
    """
    path = Path(infile)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        if not first:
            raise RuntimeError(f"Empty file: {infile}")

        charlin = (first.rstrip("\n") + " " * 13)[:13]  # A13
        if charlin[10:13].lower() == "nan":
            print(f"!!! nan entry in file !!! => {infile}")
            return []

        # "backspace": process first line again
        pending = [first] + f.readlines()

    mat = [0.0 for _ in range(resno + 1)]  # 1-based j
    reached_end = False

    for ln in pending:
        if not ln.strip():
            continue

        # Fortran 21: (1x,I4,1x,I4,1x,F6.1)
        # robust parse: try fixed slices first, then split
        s = ln.rstrip("\n")
        i = _as_int(s[1:5].strip(), 0)
        j = _as_int(s[6:10].strip(), 0)
        v = None
        if len(s) >= 11:
            v = _as_float(s[11:].strip(), 0.0)
        else:
            v = 0.0

        if j == 0:
            parts = ln.strip().split()
            if len(parts) >= 3:
                i = _as_int(parts[0], 0)
                j = _as_int(parts[1], 0)
                v = _as_float(parts[2], 0.0)

        if 1 <= j <= resno:
            mat[j] = float(v)

        if j == resno:
            reached_end = True
            break

    # Fortran assumes it hit j==resno; if not, proceed anyway with what we got
    hinges: List[int] = []
    for j in range(1, resno):
        a = mat[j]
        b = mat[j + 1]
        if (a > 0.0 and b < 0.0) or (a < 0.0 and b > 0.0):
            hinges.append(j)
            if len(hinges) >= 100:
                break

    return hinges


# ---------------------------- main pipeline ----------------------------------

def main(workdir: str = "."):
    wd = Path(workdir)

    rescale = _read_first_float(wd / "rescale")
    resno = _read_first_int(wd / "coordinates")

    file1 = "crosscorrslow1"
    file2 = "crosscorrslow2"

    alpha_lines = (wd / "alpha.cor").read_text(encoding="utf-8", errors="replace").splitlines()
    if len(alpha_lines) < resno:
        raise RuntimeError(f"alpha.cor has {len(alpha_lines)} lines, expected {resno}")

    # x,y,z,boy: [mode][res-1] mode=0..9, res=0..resno-1
    x = [[0.0 for _ in range(resno)] for _ in range(10)]
    y = [[0.0 for _ in range(resno)] for _ in range(10)]
    z = [[0.0 for _ in range(resno)] for _ in range(10)]
    boy = [[0.0 for _ in range(resno)] for _ in range(10)]

    # from alpha.cor: restyp(i), cano(i)
    restyp = [""] * resno
    cano = [0] * resno

    # from newcoordinat.mds: label, attyp, restyp, chtyp, ind, resex, xdist, ydist, zdist
    label = [""] * resno
    attyp = [""] * resno
    chtyp = [""] * resno
    ind = [0] * resno
    resex = [""] * resno
    xdist = [0.0] * resno
    ydist = [0.0] * resno
    zdist = [0.0] * resno

    # Open 1coor..10coor
    coor_fhs = [open(wd / f"{k}coor", "r", encoding="utf-8", errors="replace") for k in range(1, 11)]

    f_mds_in = open(wd / "newcoordinat.mds", "r", encoding="utf-8", errors="replace")
    f_mds_out = open(wd / "newcoordinat2.mds", "w", encoding="utf-8")
    f_len = open(wd / "anm_length", "w", encoding="utf-8")
    f_map = open(wd / "mapping.out", "w", encoding="utf-8")

    try:
        for i in range(resno):
            rt, cn = parse_alpha_line(alpha_lines[i])
            restyp[i] = rt
            cano[i] = cn

            # read coor lines for 10 modes
            for m in range(10):
                ln = coor_fhs[m].readline()
                if not ln:
                    raise RuntimeError(f"{m+1}coor ended early at residue {i+1}/{resno}")
                xi, yi, zi, bi = parse_coor_line(ln)
                x[m][i] = xi
                y[m][i] = yi
                z[m][i] = zi
                boy[m][i] = bi

            # read newcoordinat.mds line (format 100)
            mds_line = f_mds_in.readline()
            if not mds_line:
                raise RuntimeError(f"newcoordinat.mds ended early at residue {i+1}/{resno}")
            row = parse_mds100(mds_line)

            label[i] = row.label
            attyp[i] = row.attyp
            restyp[i] = row.restyp  # overwrite from file (as Fortran does)
            chtyp[i] = row.chtyp
            ind[i] = row.ind
            resex[i] = row.resex
            xdist[i] = row.x
            ydist[i] = row.y
            zdist[i] = row.z

            # Fortran: write (35,100) ... xdist,ydist,zdist  (NO occ in list!)
            f_mds_out.write(format_mds100(row, include_occ=False))

            # anm_length: write (31,99) i,cano(i),(boy(j,i),j=1,10)
            f_len.write(format_99(i + 1, cano[i], [boy[m][i] for m in range(10)]))

            # then boy squared
            for m in range(10):
                boy[m][i] = boy[m][i] ** 2

    finally:
        for fh in coor_fhs:
            fh.close()
        f_mds_in.close()
        f_mds_out.close()
        f_len.close()

    # ---------------- mapping: read slowmodes (gnmboy) ----------------
    gnmboy = read_slowmodes(wd / "slowmodes", resno=resno)  # [10][resno]

    # sums on i=8..resno-8 (Fortran inclusive). Python indices: 7..resno-9
    anmsum = [0.0] * 10
    gnmsum = [0.0] * 10
    for j in range(10):
        for i in range(7, resno - 8):
            gnmsum[j] += gnmboy[j][i]
            anmsum[j] += boy[j][i]

    # normalize
    anmboy = [[0.0 for _ in range(resno)] for _ in range(10)]
    for j in range(10):
        gden = gnmsum[j] if gnmsum[j] != 0.0 else 1.0
        aden = anmsum[j] if anmsum[j] != 0.0 else 1.0
        for i in range(resno):
            gnmboy[j][i] = gnmboy[j][i] / gden
            anmboy[j][i] = boy[j][i] / aden

    # error matrix e(1..2,1..10)
    e = [[0.0 for _ in range(10)] for _ in range(2)]
    for k in range(2):        # k=0..1 corresponds to Fortran k=1..2
        for j in range(10):   # j=0..9 corresponds to Fortran j=1..10
            acc = 0.0
            for i in range(7, resno - 8):
                acc += abs(gnmboy[k][i] - anmboy[j][i])
            e[k][j] = acc

    # Fortran selection rules (faithful translation)
    iii = 0  # Fortran iii=1
    for j in range(1, 10):  # Fortran j=2..10
        denom = e[0][iii]
        ratio = (e[0][j] / denom) if denom != 0.0 else float("inf")
        if ratio < 0.95:
            iii = j

    jjj = 0 if iii != 0 else 1  # Fortran: if iii!=1 jjj=1 else jjj=2
    for j in range(1, 10):      # Fortran j=2..10
        if j == iii:
            continue
        denom = e[1][jjj]
        ratio = (e[1][j] / denom) if denom != 0.0 else float("inf")
        if ratio < 0.95:
            jjj = j

    # mapping.out
    f_map.write(f"selection for GNM 1 : {iii+1}\n")
    f_map.write(f"selection for GNM 2 : {jjj+1}\n")
    f_map.write(" GNM 1st mode normalized parameters \n")
    f_map.write(
        "a1={:9.5f} a2={:9.5f} a3={:9.5f} a4={:9.5f} a5={:9.5f} a6={:9.5f}\n".format(
            e[0][0], e[0][1], e[0][2], e[0][3], e[0][4], e[0][5]
        )
    )
    f_map.write(" GNM 2nd mode normalized parameters \n")
    f_map.write(
        "a1={:9.5f} a2={:9.5f} a3={:9.5f} a4={:9.5f} a5={:9.5f} a6={:9.5f}\n".format(
            e[1][0], e[1][1], e[1][2], e[1][3], e[1][4], e[1][5]
        )
    )
    f_map.close()

    # Write coor*.mds12 and gnm*anmvector
    f41 = open(wd / "coor1.mds12", "w", encoding="utf-8")
    f42 = open(wd / "coor3.mds12", "w", encoding="utf-8")
    f43 = open(wd / "coor2.mds12", "w", encoding="utf-8")
    f44 = open(wd / "coor4.mds12", "w", encoding="utf-8")
    f45 = open(wd / "gnm1anmvector", "w", encoding="utf-8")
    f46 = open(wd / "gnm2anmvector", "w", encoding="utf-8")

    try:
        for i in range(resno):
            m1 = iii
            m2 = jjj

            base = AtomRow(
                label=label[i],
                attyp=attyp[i],
                restyp=restyp[i],
                chtyp=chtyp[i],
                ind=ind[i],
                resex=resex[i],
                x=xdist[i],
                y=ydist[i],
                z=zdist[i],
                occ=1.00,
            )

            # + m1
            f41.write(format_mds100(AtomRow(base.label, base.attyp, base.restyp, base.chtyp, base.ind, base.resex,
                                           base.x + x[m1][i] * rescale, base.y + y[m1][i] * rescale, base.z + z[m1][i] * rescale, 1.00),
                                    include_occ=True))
            # + m2
            f42.write(format_mds100(AtomRow(base.label, base.attyp, base.restyp, base.chtyp, base.ind, base.resex,
                                           base.x + x[m2][i] * rescale, base.y + y[m2][i] * rescale, base.z + z[m2][i] * rescale, 1.00),
                                    include_occ=True))
            # - m1
            f43.write(format_mds100(AtomRow(base.label, base.attyp, base.restyp, base.chtyp, base.ind, base.resex,
                                           base.x - x[m1][i] * rescale, base.y - y[m1][i] * rescale, base.z - z[m1][i] * rescale, 1.00),
                                    include_occ=True))
            # - m2
            f44.write(format_mds100(AtomRow(base.label, base.attyp, base.restyp, base.chtyp, base.ind, base.resex,
                                           base.x - x[m2][i] * rescale, base.y - y[m2][i] * rescale, base.z - z[m2][i] * rescale, 1.00),
                                    include_occ=True))

            # vectors (boy is already squared at this point, matching Fortran)
            f45.write(format_101(i + 1, x[m1][i], y[m1][i], z[m1][i], boy[m1][i]))
            f46.write(format_101(i + 1, x[m2][i], y[m2][i], z[m2][i], boy[m2][i]))

    finally:
        f41.close()
        f42.close()
        f43.close()
        f44.close()
        f45.close()
        f46.close()

    # ---------------- hinges from crosscorr files ----------------
    hinge1 = findhinge(wd / file1, resno=resno)
    hinge2 = findhinge(wd / file2, resno=resno)

    # Fortran 15: (3x,I4,1x,I4,A1,2x,A1)
    with open(wd / "hinges", "w", encoding="utf-8") as f:
        f.write(" ----> crosscorrelation : 1st slowest mode\n")
        for h in hinge1:
            idx = h - 1
            f.write(f"   {h:4d} {cano[idx]:4d}{resex[idx]:1s}  {chtyp[idx]:1s}\n")

        f.write(" ----> crosscorrelation : 2nd slowest mode\n")
        for h in hinge2:
            idx = h - 1
            f.write(f"   {h:4d} {cano[idx]:4d}{resex[idx]:1s}  {chtyp[idx]:1s}\n")


if __name__ == "__main__":
    main(".")
