#!/usr/bin/env python3
"""
coor2pdb.py

Fortran port of:
  program main + subroutine coor2pdb

Purpose
- Convert ANM/GNM coordinate displacement files (Xcoor-like) + base 'pdb' file
  into multi-model PDBs for animation.
- Matches the original logic:
  * Reads infile lines: i, x(i), y(i), z(i)   (1-based residue index)
  * Computes magnitude and resc scaling so average motion ~ 3 Å (with cap at 0.75/mag)
  * Writes models for modind = -modnum..modnum (modnum=3 => 7 models)
  * Applies per-residue displacement to ALL atoms in each residue
  * Skips writing 'TER' lines after last atom block (like original)

Usage
  python3 coor2pdb.py                # runs the "main" loop with the 38 mappings
  python3 coor2pdb.py --in 1coor --out 1anm.pdb
  python3 coor2pdb.py --pdb pdb --in 1coor --out 1anm.pdb --ang 3.0 --modnum 3
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# -------------------- helpers --------------------
def _read_text_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n") for ln in f]


def _write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln if ln.endswith("\n") else ln + "\n")


def _is_atom_line(line: str) -> bool:
    return line.startswith("ATOM  ")


def _is_end_line(line: str) -> bool:
    # Fortran checked dummy == "END   " or "ENDMDL"
    rec = (line[:6] if len(line) >= 6 else line).strip()
    return rec in {"END", "ENDMDL"}


def _parse_atom_fields_pdb(line: str) -> Tuple[int, str, str, str, int, str, float, float, float, float, float, str]:
    """
    Parse fields compatible with the Fortran FORMAT 2:
      A6, I5, 2X, A3, 1X, A3, 1X, A1, 1X, I3, A1, 3X, 3F8.3, F6.2, F6.2, 10X, A2

    We use fixed-width PDB columns (more robust than matching the old format exactly).
    Returns:
      atnum, attyp(4), restyp(3), chain(1), resseq(int), icode(1),
      x,y,z, occup, bfac, element(2)
    """
    # PDB standard columns (1-based):
    # 1-6 record, 7-11 serial, 13-16 name, 18-20 resName, 22 chainID,
    # 23-26 resSeq, 27 iCode, 31-38 x, 39-46 y, 47-54 z, 55-60 occ, 61-66 bfac, 77-78 element
    atnum = int(line[6:11].strip() or "0")
    attyp = (line[12:16] if len(line) >= 16 else "").ljust(4)[:4]
    restyp = (line[17:20] if len(line) >= 20 else "").strip().ljust(3)[:3]
    chain = (line[21:22] if len(line) >= 22 else " ").strip() or " "
    resseq = int((line[22:26] if len(line) >= 26 else "").strip() or "0")
    icode = (line[26:27] if len(line) >= 27 else " ").strip() or " "
    x = float((line[30:38] if len(line) >= 38 else "0").strip() or "0")
    y = float((line[38:46] if len(line) >= 46 else "0").strip() or "0")
    z = float((line[46:54] if len(line) >= 54 else "0").strip() or "0")
    occup = float((line[54:60] if len(line) >= 60 else "1.00").strip() or "1.00")
    bfac = float((line[60:66] if len(line) >= 66 else "0.00").strip() or "0.00")
    element = (line[76:78] if len(line) >= 78 else "").strip().rjust(2)
    return atnum, attyp, restyp, chain, resseq, icode, x, y, z, occup, bfac, element


def _format_atom_line(
    atnum: int,
    attyp4: str,
    restyp3: str,
    chain1: str,
    resseq: int,
    icode1: str,
    x: float,
    y: float,
    z: float,
    occup: float,
    bfac: float,
    element2: str,
) -> str:
    """
    Write PDB-like ATOM record. We keep it close to standard PDB formatting.
    """
    # Ensure widths
    attyp4 = attyp4.ljust(4)[:4]
    restyp3 = restyp3.ljust(3)[:3]
    chain1 = (chain1 or " ")[:1]
    icode1 = (icode1 or " ")[:1]
    element2 = (element2 or "").strip().rjust(2)[:2]

    # This mirrors typical PDB alignment.
    # Note: we don't preserve original altLoc; the Fortran didn't either.
    return (
        f"ATOM  {atnum:5d} {attyp4}{' '}{restyp3} {chain1}{resseq:4d}{icode1}"
        f"   {x:8.3f}{y:8.3f}{z:8.3f}{occup:6.2f}{bfac:6.2f}          {element2}"
    )


# -------------------- core logic --------------------
@dataclass
class CoorData:
    # 1-based arrays (index 1..resnum)
    x: List[float]
    y: List[float]
    z: List[float]
    mag: List[float]
    resc: List[float]
    resnum: int


def read_coor_file(path: str, resmax: int = 5000) -> CoorData:
    """
    Reads lines like:
      i  x(i)  y(i)  z(i)
    Stops at EOF, sets resnum to the last i read (like Fortran).
    """
    x = [0.0] * (resmax + 1)
    y = [0.0] * (resmax + 1)
    z = [0.0] * (resmax + 1)
    mag = [0.0] * (resmax + 1)

    last_i = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            i = int(float(parts[0]))
            if i < 1 or i > resmax:
                continue
            xi = float(parts[1]); yi = float(parts[2]); zi = float(parts[3])
            x[i] = xi; y[i] = yi; z[i] = zi
            mag[i] = (xi * xi + yi * yi + zi * zi) ** 0.5
            last_i = i

    if last_i <= 0:
        raise RuntimeError(f"Empty/invalid coor file: {path}")

    resnum = last_i
    resc = [0.0] * (resmax + 1)
    return CoorData(x=x, y=y, z=z, mag=mag, resc=resc, resnum=resnum)


def compute_rescales(data: CoorData, ang: float = 3.0) -> None:
    """
    Fortran:
      magsum = avg_{i=10..resnum-10} mag(i)  (but then divided by resnum, not count!)
      rescale = ang / magsum
      resc(i) = 0.75/mag(i) if rescale*mag(i) >= 3 else rescale
    """
    resnum = data.resnum
    if resnum < 25:
        # still do something reasonable; keep the Fortran formula but avoid empty ranges
        i0, i1 = 1, resnum
    else:
        i0, i1 = 10, resnum - 10

    magsum = 0.0
    for i in range(i0, i1 + 1):
        magsum += data.mag[i]

    # Keep the Fortran division by real(resnum), not (i1-i0+1)
    magsum = magsum / float(resnum) if resnum > 0 else 1.0
    if magsum == 0.0:
        magsum = 1e-12

    rescale = ang / magsum

    for i in range(1, resnum + 1):
        mi = data.mag[i]
        if mi <= 0.0:
            data.resc[i] = 0.0
            continue
        if rescale * mi >= 3.0:
            data.resc[i] = 0.75 / mi
        else:
            data.resc[i] = rescale


def coor2pdb(
    infile: str,
    outfile: str,
    pdb_path: str = "pdb",
    ang: float = 3.0,
    modnum: int = 3,
    resmax: int = 5000,
) -> None:
    """
    Port of subroutine coor2pdb(infile,outfile).
    """
    data = read_coor_file(infile, resmax=resmax)
    compute_rescales(data, ang=ang)

    pdb_lines = _read_text_lines(pdb_path)
    if not pdb_lines:
        raise RuntimeError(f"Base PDB file is empty: {pdb_path}")

    # Find first ATOM line: Fortran copied header lines until first ATOM
    out_lines: List[str] = []
    matomline = None
    for idx, ln in enumerate(pdb_lines):
        if _is_atom_line(ln):
            matomline = idx
            break
        out_lines.append(ln)

    if matomline is None:
        raise RuntimeError(f"No ATOM lines found in base PDB: {pdb_path}")

    # Determine last atom line position while processing (Fortran set lastatomline inside)
    lastatomline = matomline

    # Fortran: modind = (-1)*modnum; loop while modind <= modnum
    # Model number written: modind + modnum + 1  => 1..(2*modnum+1)
    for modind in range(-modnum, modnum + 1):
        pdbi = matomline
        i_res = 0          # index into coor residues (1..resnum)
        ii_resseq = None   # last residue seq encountered

        out_lines.append(f"MODEL {modind + modnum + 1:2d}")

        while pdbi < len(pdb_lines):
            ln = pdb_lines[pdbi]
            rec6 = ln[:6]

            if _is_end_line(ln):
                break

            if rec6 == "ATOM  ":
                lastatomline = pdbi
                atnum, attyp, restyp, chain, resseq, icode, xx, yy, zz, occ, bfac, elem = _parse_atom_fields_pdb(ln)

                if ii_resseq is None or resseq != ii_resseq:
                    ii_resseq = resseq
                    i_res += 1

                if i_res > data.resnum:
                    # If base PDB has more residues than coor file, stop displacing further residues
                    rscaled = 0.0
                    dx = dy = dz = 0.0
                else:
                    rscaled = data.resc[i_res] * (float(modind) / float(modnum))
                    dx = data.x[i_res] * rscaled
                    dy = data.y[i_res] * rscaled
                    dz = data.z[i_res] * rscaled

                xx1 = xx + dx
                yy1 = yy + dy
                zz1 = zz + dz

                out_lines.append(_format_atom_line(
                    atnum=atnum,
                    attyp4=attyp,
                    restyp3=restyp,
                    chain1=chain,
                    resseq=resseq,
                    icode1=icode,
                    x=xx1, y=yy1, z=zz1,
                    occup=occ,
                    bfac=bfac,
                    element2=elem,
                ))
            else:
                # In Fortran, once it reached ATOM region, it did not copy other records inside
                # the model loop except it stops at END/ENDMDL. We'll ignore non-ATOM lines here.
                pass

            pdbi += 1

        out_lines.append("ENDMDL")

    # After model loop: if modind == modnum it wrote END; our loop always ends at modnum so write END
    out_lines.append("END")

    # Write remaining lines after last atom line, excluding TER (like Fortran)
    for ln in pdb_lines[lastatomline + 1:]:
        if ln.startswith("TER"):
            continue
        out_lines.append(ln)

    _write_lines(outfile, out_lines)


# -------------------- "main" program --------------------
def _default_mapping() -> Tuple[List[str], List[str]]:
    infile = [""] * 38
    outfile = [""] * 38

    infile[0] = "gnm1anmvector"
    infile[1] = "gnm2anmvector"
    for k in range(3, 39):  # 3..38 in Fortran; 1-based
        infile[k - 1] = f"{k-2}coor"  # 1coor..36coor

    outfile[0] = "mod1"
    outfile[1] = "mod2"
    for k in range(3, 39):
        outfile[k - 1] = f"{k-2}anm.pdb"  # 1anm.pdb..36anm.pdb

    return infile, outfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="pdb", help="Base PDB file path (default: pdb)")
    ap.add_argument("--in", dest="infile", default=None, help="Single input coor file to convert")
    ap.add_argument("--out", dest="outfile", default=None, help="Single output PDB file to write")
    ap.add_argument("--ang", type=float, default=3.0, help="Target average motion angstrom (default 3.0)")
    ap.add_argument("--modnum", type=int, default=3, help="Half-model count (default 3 -> 7 models)")
    args = ap.parse_args()

    if args.infile and args.outfile:
        coor2pdb(args.infile, args.outfile, pdb_path=args.pdb, ang=args.ang, modnum=args.modnum)
        return

    infile_list, outfile_list = _default_mapping()
    for inf, outf in zip(infile_list, outfile_list):
        if not os.path.exists(inf):
            # match the spirit of Fortran (would crash); here we skip with a clear message
            raise RuntimeError(f"Missing input file: {inf}")
        coor2pdb(inf, outf, pdb_path=args.pdb, ang=args.ang, modnum=args.modnum)


if __name__ == "__main__":
    main()
