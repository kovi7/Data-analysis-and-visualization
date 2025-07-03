import sys
import os

def parse_ss3_file(filename):
    """
    Parses a pseudo-FASTA file with secondary structure predictions.
    Returns counts of helix (H), strand (E), coil (-) and total residues.
    """
    helix = 0
    strand = 0
    coil = 0
    total = 0

    with open(filename) as f:
        lines = f.readlines()
        # File format: >header, sequence, ss3, score
        for i in range(0, len(lines), 4):
            if i+2 >= len(lines):
                break  # skip incomplete record
            ss_line = lines[i+2].strip()
            helix += ss_line.count('H')
            strand += ss_line.count('E')
            coil += ss_line.count('-')
            total += len(ss_line)
    return helix, strand, coil, total

def print_and_save_percentages(label, helix, strand, coil, total, summary_file):
    helix_pct = helix/total*100 if total else 0
    strand_pct = strand/total*100 if total else 0
    coil_pct = coil/total*100 if total else 0
    summary_line = (f"{os.path.basename(label)}| Helix: {helix_pct:6.2f}% | "
                    f"Strand: {strand_pct:6.2f}% | Coil: {coil_pct:6.2f}% | "
                    f"Total residues: {total}\n")
    print(summary_line.strip())
    #summary file
    with open(summary_file, "a") as f:
        f.write(summary_line)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 calc_ss3_percent.py file1_ss3.fasta [file2_ss3.fasta ...]")
        sys.exit(1)

    for fname in sys.argv[1:]:
        helix, strand, coil, total = parse_ss3_file(fname)
        print_and_save_percentages(fname, helix, strand, coil, total, 'protbert_data/protbert_summary.txt')
