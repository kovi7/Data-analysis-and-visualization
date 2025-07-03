import sys

def split_fasta(input_file, num_sequences):
    with open(f'../raw_data/{input_file}', 'r') as infile:
        seq_count = 0
        part = 0
        outfile = None

        for line in infile:
            if line.startswith(">"):
                seq_count += 1
                if seq_count > num_sequences:
                    outfile.close()
                    part += 1
                    seq_count = 1

                if seq_count == 1:
                    #num = int(part*(num_sequences/1000))
                    outfile = open(f"../split_data/{input_file.split('.')[0]}_{part}k.fasta", "w")

            outfile.write(line)

        if outfile:
            outfile.close()

if __name__ == "__main__":
    input_file = sys.argv[1]
    num_sequences = int(sys.argv[2]) * 1000
    if num_sequences > 1000000:
        print("ERROR: Maximum allowed number of sequences per file is 10,00,000.")
        sys.exit(1)
    split_fasta(input_file, num_sequences)
