import sys

def normalize_fasta(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        seq = ""
        for line in infile:
            if line.startswith(">"):
                if seq:
                    outfile.write(seq + "\n")
                outfile.write(line)
                seq = ""
            else:
                seq += line.strip()
        if seq:
            outfile.write(seq + "\n")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    normalize_fasta(input_file, output_file)
