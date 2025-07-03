import os
import sys
import tempfile
import time

def run_iupred_short(input_file, iupred_type='long'):
    output_file = f"iupred_data/{os.path.splitext(os.path.basename(input_file))[0]}_iupred_long.fasta"
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        start_time = time.time()
        for line in infile:
            if line.startswith(">"):
                header = line.strip()
                sequence = next(infile).strip()

                # Create temporary input file
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_input:
                    tmp_input.write(f"{header}\n{sequence}\n")
                    tmp_input_path = tmp_input.name

                try:
                    # Run IUPred2A with short prediction and data directory
                    cmd = f"python3 iupred2a/iupred2a.py {tmp_input_path} {iupred_type}"
                    output = os.popen(cmd).read()
                    
                    # Parse results
                    pred_lines = [l for l in output.split('\n') if l and not l.startswith('#')]
                    if len(pred_lines) < len(sequence):
                        print(f"Error processing {header}: insufficient output")
                        continue
                    pred_scores = [float(line.split()[2]) for line in pred_lines[:len(sequence)]]
                    pred_bin = ''.join(['D' if score >= 0.5 else '-' for score in pred_scores])
                    accuracy = ''.join(['8' if score >= 0.8 else '6' for score in pred_scores])
                    
                    outfile.write(f"{header}\n{sequence}\n{pred_bin}\n{accuracy}\n\n")
                finally:
                    # Cleanup temporary files
                    os.remove(tmp_input_path)
        elapsed = time.time() - start_time
        basename = os.path.basename(sys.argv[1])
        with open("iupred_data/iupred_times.txt", "a") as f:
            f.write(f"{basename} cpu {elapsed:.2f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 iupred_runner.py <input.fasta>")
        sys.exit(1)
    
    # Verify data directory exists
    if not os.path.exists('.'):
        print("Error: IUPred2A data directory not found at '.'")
        print("Download data files from https://iupred2a.elte.hu/download and extract to ./data/")
        sys.exit(1)
    
    run_iupred_short(sys.argv[1], iupred_type='long')
