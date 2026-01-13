# hingeaa.py
import re
import sys


def _read_hinges(hingeain_path: str):
    """
    C kodundaki ilk pass: 'Hinge residues' satır(lar)ındaki >0 integerları sırayla toplar.
    """
    hinges = []
    with open(hingeain_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # C: "Hinge residues" ile başlayan satırlarda token token gezip sayı topluyor
            if line.startswith("Hinge residues"):
                nums = re.findall(r"\d+", line)
                for s in nums:
                    v = int(s)
                    if v > 0:
                        hinges.append(v)
    return hinges


def _read_aa_from_coordinates(coord_path: str):
    """
    C kodu:
      fscanf(coord,"%d",&dummy);  # N
      aa[j] = line[35],line[36],line[37]
    """
    with open(coord_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        if not first:
            raise RuntimeError("coordinates file is empty.")
        parts = first.strip().split()
        if not parts:
            raise RuntimeError("coordinates first line does not contain N.")
        n = int(parts[0])

        aa = []
        for _ in range(n):
            line = f.readline()
            if not line:
                # dosya beklenenden erken biterse
                break
            # C: 35,36,37 karakterleri
            code = line[35:38] if len(line) >= 38 else ""
            code = code.rstrip("\n")
            if len(code) < 3:
                code = (code + "???")[:3]
            aa.append(code)
    return aa  # length <= n


def hingeaa(hingeain_path: str, coord_path: str, hingeout_path: str, echo: bool = True):
    hinges = _read_hinges(hingeain_path)
    aa = _read_aa_from_coordinates(coord_path)

    # C kodu ikinci pass: dosyayı baştan okur, hinge pointer j ile sırayla eşleştirir
    j = 0
    with open(hingeain_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(hingeout_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.startswith("Hinge residues"):
                fout.write(line)
                if echo:
                    sys.stdout.write(line)
                continue

            fout.write("Hinge residues: ")
            if echo:
                sys.stdout.write("Hinge residues: ")

            nums = [int(x) for x in re.findall(r"\d+", line)]
            for dummy in nums:
                if dummy > 0 and j < len(hinges) and hinges[j] == dummy:
                    # C: aa[hinge[j]] ile indexliyor (aynen koruyoruz)
                    if 0 <= hinges[j] < len(aa):
                        code = aa[hinges[j]]
                    else:
                        code = "???"
                    chunk = f"{hinges[j]} {code}  "
                    fout.write(chunk)
                    if echo:
                        sys.stdout.write(chunk)
                    j += 1

            fout.write("\n")
            if echo:
                sys.stdout.write("\n")


def main():
    # Varsayılanlar C binary ile aynı isimler
    hingeain = "hingeain"
    coordinates = "coordinates"
    hingeout = "hingeout"

    if len(sys.argv) >= 2:
        hingeain = sys.argv[1]
    if len(sys.argv) >= 3:
        coordinates = sys.argv[2]
    if len(sys.argv) >= 4:
        hingeout = sys.argv[3]

    hingeaa(hingeain, coordinates, hingeout, echo=True)


if __name__ == "__main__":
    main()
