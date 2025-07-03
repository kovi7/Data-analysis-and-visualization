import sys, os

def calculate_disorder(file_path, summary_path):
    with open(file_path) as f:
        d, total = 0, 0
        for line in f:
            if not line.startswith(">") and set(line.strip()) <= {"D", "-"}:
                d += line.count("D")
                total += len(line.strip())
    with open(summary_path, 'a') as out:
        out.write(f"{os.path.basename(file_path)}: {round(d * 100 / total,1)}% disorder\n")
    print(f"Disorder: {round(d * 100 / total,1)}%")

if __name__ == "__main__":
     summary = sys.argv[2] if len(sys.argv) > 2 else "../iupred_data/disorder_summary.txt"
     calculate_disorder(sys.argv[1], summary)
