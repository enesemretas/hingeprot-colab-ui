#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Iterator, Tuple, Optional, Dict, TextIO


def _iter_modeent_records(path: str) -> Iterator[Tuple[str, int, int]]:
    """
    Fortran format 51: (A4,17x,A1,I4,34x,I3)
    Columns (1-based):
      1-4   : 'ATOM'
      22    : chainr (A1)
      23-26 : ir (I4)
      61-63 : bfactr (I3)
    Fortran: read(2,*) -> first line skipped
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # skip first line (read(2,*))
        _ = f.readline()

        for line in f:
            if len(line) < 63:
                continue
            dummy = line[0:4]
            if dummy.strip() != "ATOM":
                continue
            chainr = line[21:22]  # col 22
            try:
                ir = int(line[22:26])          # col 23-26
                bfactr = int(line[60:63])      # col 61-63
            except Exception:
                continue
            yield (chainr, ir, bfactr)


def _pdb_atom_stream(path: str) -> Iterator[str]:
    """
    Fortran:
      open(unit=1, file="pdb")
      read until '(A4)' == 'ATOM'
      then read with format 50 line-by-line; stop when dummy != 'ATOM'
    We will:
      - skip until line starts with 'ATOM'
      - then yield subsequent lines until a non-ATOM line appears (END/EOF breaks)
      - TER lines are yielded too (caller decides to skip)
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # skip until ATOM in first 4 chars
        for line in f:
            if line[:4] == "ATOM":
                # first ATOM line found; yield it and then continue
                yield line.rstrip("\n")
                break
        else:
            return  # no ATOM at all

        for line in f:
            if line[:4] != "ATOM" and not line.startswith("TER"):
                break
            yield line.rstrip("\n")


def _parse_pdb_fields(line: str) -> Optional[Tuple[str, str, int]]:
    """
    Fortran format 50: (A21,A1,I4,A34,F6.2,A24)
    chain: col 22
    resnum: col 23-26 (I4)
    """
    if len(line) < 26:
        return None
    chain = line[21:22]
    try:
        resnum = int(line[22:26])
    except Exception:
        return None
    return (chain, chain, resnum)


def _format_like_fortran_50(line: str) -> str:
    """
    Reconstruct line using Fortran 50 blocks:
      st1(21) + chain(1) + resnum(I4) + st2(34) + bfact(F6.2) + st3(24)
    We take from original columns where possible.
    If parsing bfact fails, we keep original line as-is.
    """
    try:
        st1 = (line[0:21]).ljust(21)
        chain = (line[21:22] if len(line) > 21 else " ")
        resnum = int(line[22:26])
        st2 = (line[26:60] if len(line) >= 60 else line[26:].ljust(34)).ljust(34)[:34]
        bfact_str = (line[60:66] if len(line) >= 66 else "").strip()
        bfact = float(bfact_str) if bfact_str else 0.0
        st3 = (line[66:90] if len(line) >= 90 else line[66:].ljust(24)).ljust(24)[:24]
        return f"{st1}{chain}{resnum:4d}{st2}{bfact:6.2f}{st3}"
    except Exception:
        return line  # fallback


def split(start: int, infile: str, pdb_file: str = "pdb") -> None:
    """
    Python equivalent of:
      call split(start, infile)
    Writes to fort.<unit> where unit = start + (bfactr/5) using integer division.
    """
    # modeent iterator
    rec_iter = _iter_modeent_records(infile)
    current = next(rec_iter, None)  # (chainr, ir, bfactr)

    if current is None:
        return

    # open output files lazily and keep handles
    outs: Dict[int, TextIO] = {}

    def _get_out(unit: int) -> TextIO:
        if unit not in outs:
            fname = f"fort.{unit}"
            outs[unit] = open(fname, "w", encoding="utf-8")
        return outs[unit]

    try:
        for raw in _pdb_atom_stream(pdb_file):
            if raw.startswith("TER"):
                continue
            if raw[:4] != "ATOM":
                break

            parsed = _parse_pdb_fields(raw)
            if parsed is None:
                continue
            chain = parsed[0]
            resnum = parsed[2]

            # advance modeent until match (Fortran: backspace pdb + read next modeent)
            while current is not None:
                chainr, ir, bfactr = current
                if (chainr == chain) and ((resnum % 1000) == (ir % 1000)):
                    unit = start + (bfactr // 5)
                    out = _get_out(unit)
                    out.write(_format_like_fortran_50(raw) + "\n")
                    # IMPORTANT: do NOT advance modeent here (Fortran keeps same record
                    # for all atoms of the same residue)
                    break
                else:
                    current = next(rec_iter, None)

            if current is None:
                break
    finally:
        for fh in outs.values():
            try:
                fh.close()
            except Exception:
                pass


def main():
    # Fortran main:
    # start1=10; start2=20; infile1='modeent1'; infile2='modeent2'
    if os.path.exists("modeent1"):
        split(10, "modeent1", pdb_file="pdb")
    if os.path.exists("modeent2"):
        split(20, "modeent2", pdb_file="pdb")


if __name__ == "__main__":
    main()
