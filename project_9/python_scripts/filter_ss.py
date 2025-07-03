import sys

def parse_fasta(filepath):
    with open(filepath) as f:
        lines = [line.strip() for line in f]
    entries = []
    for i in range(0, len(lines), 4):
        header = lines[i]
        seq = lines[i + 1]
        ss = lines[i + 2]
        entries.append((header, seq, ss))
    return entries

def calc_score(ss, w_h, w_c, w_e):
    return (w_h * ss.count('H') + w_c * ss.count('-') + w_e * ss.count('E')) / len(ss)

def filter_proteins(entries, min_len, min_h, max_e, w_h, w_c, w_e):
    filtered = []
    for header, seq, ss in entries:
        if len(seq) < min_len:
            continue
        percent_h = 100 * ss.count('H') / len(seq)
        percent_e = 100 * ss.count('E') / len(seq)
        if percent_h >= min_h and percent_e <= max_e:
            score = calc_score(ss, w_h, w_c, w_e)
            header_with_score = header + f"|SHS-score={score:.2f}"
            filtered.append((score, header_with_score, seq, ss))
    return sorted(filtered, key=lambda x: x[0], reverse=True)

def remove_duplicates(filtered):
    seen = set()
    unique = []
    for score, header, seq, ss in filtered:
        if seq not in seen:
            seen.add(seq)
            unique.append((score, header, seq, ss))
    return unique

def main():
    infile = sys.argv[1]
    top_n = int(sys.argv[2])
    min_len = int(sys.argv[3])
    min_h = int(sys.argv[4])
    max_e = int(sys.argv[5])
    w_h = int(sys.argv[6])
    w_c = int(sys.argv[7])
    w_e = int(sys.argv[8])
    remove_dups = int(sys.argv[9])

    entries = parse_fasta(infile)
    filtered = filter_proteins(entries, min_len, min_h, max_e, w_h, w_c, w_e)

    if remove_dups:
        filtered = remove_duplicates(filtered)

    outname = infile.replace('.fasta', f"_SHS_{top_n}.fasta")
    with open(outname, 'w') as out:
        for entry in filtered[:top_n]:
            out.write(f"{entry[1]}\n{entry[2]}\n{entry[3]}\n\n")

    print(f"Saved top {min(len(filtered), top_n)} entries to {outname}")

if __name__ == "__main__":
    main()
