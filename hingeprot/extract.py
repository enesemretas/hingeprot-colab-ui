# extract.py  (converted from extract.f / program hingebend)
# Usage (in a run folder):
#   python3 extract.py
#
# Expects files (same as Fortran):
#   rescale
#   coordinates
#   alpha.cor
#   1coor .. 10coor
#   newcoordinat.mds
#   slowmodes
#   crosscorrslow1
#   crosscorrslow2
#
# Produces:
#   anm_length
#   newcoordinat2.mds
#   mapping.out
#   coor1.mds12 coor2.mds12 coor3.mds12 coor4.mds12
#   gnm1anmvector gnm2anmvector
#   hinges

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# ---------------------------- fixed-width helpers ----------------------------

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
# Fortran formats (approx):
# 97  format (17x,A3,2x,I4)     -> alpha.cor: residue type + cano
# 98  format (4x,4(3x,F8.5))    -> coor files: x y z boy
# 100 format (a4,8x,a4,1x,a3,1x,a1,i4,a1,f11.3,f8.3,f8.3,1x,f5.2)
# 101 format (I4,4(3x,F8.5))    -> gnm1anmvector lines: i x y z boy
# 16  format(4x,10(4x,F8.5))    -> slowmodes rows: 10 floats


@dataclass
class AtomRow:
    label: str      # a4
    attyp: str      # a4
    restyp: str     # a3
    chtyp: str      # a1
    ind: int        # i4
    resex: str      # a1
    x: float        # f11.3
    y: float        # f8.3
    z: float        # f8.3
    occ: float      # f5.2


def parse_alpha_line(line: str) -> Tuple[str, int]:
    # 17x,A3,2x,I4
    restyp = line[17:20].strip()
    cano = _as_int(line[22:26].strip(), 0)
    return restyp, cano


def parse_coor_line(line: str) -> Tuple[float, float, float, float]:
    # 4x then 4 numbers with 3x padding; easiest: split
    parts = line.strip().split()
    if len(parts) < 4:
        # try fixed slices: best-effort
        return (_as_float(line[7:15]), _as_float(line[18:26]), _as_float(line[29:37]), _as_float(line[40:48]))
    return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))


def parse_mds100(line: str) -> AtomRow:
    # 100 format: a4,8x,a4,1x,a3,1x,a1,i4,a1,f11.3,f8.3,f8.3,1x,f5.2
    label = line[0:4].strip()
    attyp = line[12:16].strip()
    restyp = line[17:20].strip()
    chtyp = line[21:22].strip()
    ind = _as_int(line[22:26].strip(), 0)
    resex = line[26:27].strip()
    x = _as_float(line[27:38].strip(), 0.0)
    y = _as_float(line[38:46].strip(), 0.0)
    z = _as_float(line[46:54].strip(), 0.0)
    occ = _as_float(line[55:60].strip(), 0.0)
    return AtomRow(label, attyp, restyp, chtyp, ind, resex, x, y, z, occ)


def format_mds100(row: AtomRow) -> str:
    # mimic Fortran spacing closely enough for downstream usage
    # label(a4) + 8x + attyp(a4) + 1x + restyp(a3) + 1x + chtyp(a1) + ind(i4) + resex(a1) +
    # x(f11.3) y(f8.3) z(f8.3) + 1x + occ(f5.2)
    return (
        f"{row.label:<4}"
        + " " * 8
        + f"{row.attyp:<4} "
        + f"{row.restyp:<3} "
        + f"{row.chtyp:<1}"
        + f"{row.ind:4d}"
        + f"{row.resex:<1}"
        + f"{row.x:11.3f}{row.y:8.3f}{row.z:8.3f}"
        + f" {row.occ:5.2f}"
        + "\n"
    )


def format_101(i: int, x: float, y: float, z: float, boy: float) -> str:
    # (I4,4(3x,F8.5))
    return f"{i:4d}{x:11.5f}{y:11.5f}{z:11.5f}{boy:11.5f}\n"


def format_99(i: int, cano: int, boys: List[float]) -> str:
    # (I4,2x,I4,10(3x,F8.5))
    s = f"{i:4d}  {cano:4d}"
    for v in boys:
        s += f"{v:11.5f}"
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
                # slowmodes is fixed-format; try padding
                vals = vals + [0.0] * (10 - len(vals))
            for j in range(10):
                gnmboy[j][i] = vals[j]
    return gnmboy


# ---------------------------- hinge finder -----------------------------------

def findhinge(infile: str | Path, resno: int) -> List[int]:
    """
    Mirrors Fortran subroutine findhinge:
    - Reads crosscorr file lines: i j value
    - Stores value per j
    - Hinge positions where sign changes between j and j+1
    """
    path = Path(infile)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        if not first:
            raise RuntimeError(f"Empty file: {infile}")
        # Fortran checks charlin(11:13) == 'nan'
        if len(first) >= 13 and first[10:13].lower() == "nan":
            print(f"!!! nan entry in file !!! => {infile}")
            return []
        # go back: easiest is to include the first line back
        lines = [first] + f.readlines()

    mat = [0.0 for _ in range(resno + 1)]  # 1-based j
    last_j = 0
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) < 3:
            continue
        i = _as_int(parts[0], 0)
        j = _as_int(parts[1], 0)
        v = _as_float(parts[2], 0.0)
        if 1 <= j <= resno:
            mat[j] = v
            last_j = max(last_j, j)

    if last_j < resno:
        # Fortran expects j==resno at end; still proceed with what we have
        pass

    hinges: List[int] = []
    for j in range(1, resno):
        a = mat[j]
        b = mat[j + 1]
        if (a > 0 and b < 0) or (a < 0 and b > 0):
            hinges.append(j)
            if len(hinges) >= 100:
                break
    return hinges


# ---------------------------- main pipeline ----------------------------------

def main(workdir: str = "."):
    wd = Path(workdir)

    rescale = _read_first_float(wd / "rescale")

    file1 = "crosscorrslow1"
    file2 = "crosscorrslow2"

    # coordinates file first integer is resno
    resno = _read_first_int(wd / "coordinates")

    # Read per-residue data
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
    coor_fhs = []
    for k in range(1, 11):
        coor_fhs.append(open(wd / f"{k}coor", "r", encoding="utf-8", errors="replace"))

    # Open newcoordinat.mds and newcoordinat2.mds outputs + anm_length
    f_mds_in = open(wd / "newcoordinat.mds", "r", encoding="utf-8", errors="replace")
    f_mds_out = open(wd / "newcoordinat2.mds", "w", encoding="utf-8")
    f_len = open(wd / "anm_length", "w", encoding="utf-8")

    # Fortran also writes mapping.out later; we do too
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

            # read & copy newcoordinat.mds line
            mds_line = f_mds_in.readline()
            if not mds_line:
                raise RuntimeError(f"newcoordinat.mds ended early at residue {i+1}/{resno}")
            row = parse_mds100(mds_line)

            label[i] = row.label
            attyp[i] = row.attyp
            restyp[i] = row.restyp  # overwrite with file's restyp (matches Fortran read)
            chtyp[i] = row.chtyp
            ind[i] = row.ind
            resex[i] = row.resex
            xdist[i] = row.x
            ydist[i] = row.y
            zdist[i] = row.z

            f_mds_out.write(format_mds100(row))

            # Fortran: write (31,99) i,cano(i),(boy(j,i),j=1,10)
            f_len.write(format_99(i + 1, cano[i], [boy[m][i] for m in range(10)]))

            # Fortran squares boy after writing
            for m in range(10):
                boy[m][i] = boy[m][i] ** 2

            # Fortran prints i
            # print(i+1)

    finally:
        for fh in coor_fhs:
            fh.close()
        f_mds_in.close()
        f_mds_out.close()
        f_len.close()

    # ---------------- mapping: read slowmodes (gnmboy) ----------------
    gnmboy = read_slowmodes(wd / "slowmodes", resno=resno)  # [10][resno]

    # sums on i=8..resno-8 (Fortran is 1-based: 8 to resno-8)
    anmsum = [0.0] * 10
    gnmsum = [0.0] * 10
    for j in range(10):
        for i in range(7, resno - 8):  # 0-based equivalent of 8..resno-8
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
    for k in range(2):  # k=0->GNM1, k=1->GNM2
        for j in range(10):  # compare with ANM mode j
            acc = 0.0
            for i in range(7, resno - 8):
                acc += abs(gnmboy[k][i] - anmboy[j][i])
            e[k][j] = acc

    # pick iii (for GNM1)
    iii = 0
    for j in range(1, 10):
        if e[0][iii] == 0.0:
            continue
        if (e[0][j] / e[0][iii]) < 0.95:
            iii = j

    # pick jjj (for GNM2) ensuring jjj != iii
    jjj = 0 if iii != 0 else 1
    for j in range(1, 10):
        if j == iii:
            continue
        if e[1][jjj] == 0.0:
            continue
        if (e[1][j] / e[1][jjj]) < 0.95:
            jjj = j

    # Fortran prints 1-based indices
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
            # Fortran uses xdist +/- x(mode,i)*rescale etc.
            row_base = AtomRow(
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

            # mode indices are 0-based in Python
            m1 = iii
            m2 = jjj

            # + m1
            r = row_base
            f41.write(format_mds100(AtomRow(r.label, r.attyp, r.restyp, r.chtyp, r.ind, r.resex,
                                           r.x + x[m1][i] * rescale, r.y + y[m1][i] * rescale, r.z + z[m1][i] * rescale, 1.00)))
            # + m2
            f42.write(format_mds100(AtomRow(r.label, r.attyp, r.restyp, r.chtyp, r.ind, r.resex,
                                           r.x + x[m2][i] * rescale, r.y + y[m2][i] * rescale, r.z + z[m2][i] * rescale, 1.00)))
            # - m1
            f43.write(format_mds100(AtomRow(r.label, r.attyp, r.restyp, r.chtyp, r.ind, r.resex,
                                           r.x - x[m1][i] * rescale, r.y - y[m1][i] * rescale, r.z - z[m1][i] * rescale, 1.00)))
            # - m2
            f44.write(format_mds100(AtomRow(r.label, r.attyp, r.restyp, r.chtyp, r.ind, r.resex,
                                           r.x - x[m2][i] * rescale, r.y - y[m2][i] * rescale, r.z - z[m2][i] * rescale, 1.00)))

            # vectors (boy already squared)
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

    # Fortran: 15 format (3x,I4,1x,I4,A1,2x,A1)
    # It prints: hinge_index, cano(hinge), resex(hinge), chtyp(hinge)
    with open(wd / "hinges", "w", encoding="utf-8") as f:
        f.write("----> crosscorrelation : 1st slowest mode\n")
        for h in hinge1:
            # h is 1-based in Fortran; our hinge list holds 1..resno-1 already
            idx = h - 1
            f.write(f"   {h:4d} {cano[idx]:4d}{resex[idx]:1s}  {chtyp[idx]:1s}\n")

        f.write("----> crosscorrelation : 2nd slowest mode\n")
        for h in hinge2:
            idx = h - 1
            f.write(f"   {h:4d} {cano[idx]:4d}{resex[idx]:1s}  {chtyp[idx]:1s}\n")


if __name__ == "__main__":
    main(".")
