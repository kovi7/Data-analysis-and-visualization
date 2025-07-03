import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import sys
import os
import time

def predict_on_device(input_file, output_file, device_type='cpu'):

    # Set cpu/gpu
    device = 0 if device_type == 'gpu' and torch.cuda.is_available() else -1

    # model
    pipeline = TokenClassificationPipeline(
    	model=AutoModelForTokenClassification.from_pretrained("Rostlab/prot_bert_bfd_ss3"),
    	tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_ss3", skip_special_tokens=True),
    	device=device
    )
    #start time
    start_time = time.time()
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        sequence = ""
        header = ""
        for line in infile:
            line = line.strip()
            if line.startswith(">"):  # Header line
                if sequence:
                    # Process the previous sequence
                    process_sequence(sequence, header, pipeline, outfile)
                header = line
                sequence = ""
            else:
                sequence += line

        # Process the last sequence
        if sequence:
            process_sequence(sequence, header, pipeline, outfile)

    end_time = time.time()  # End measuring time
    elapsed_time = end_time - start_time
    print(f"Time taken for inference (excluding model loading): {elapsed_time:.2f} seconds")
    with open("protbert_data/protbert_times.txt", "a") as time_file:
        base_name = os.path.basename(input_file)
        time_file.write(f"{base_name}, {device_type}, {elapsed_time:.2f}\n")
def process_sequence(sequence, header, pipeline, outfile):
    # Make prediction
    processed_seq = " ".join(sequence)
    prediction = pipeline(processed_seq)

   # Check if the structure exists in the prediction
    structure = ''
    score = ''
    for pred in prediction:
        if 'entity' in pred:
            structure += pred['entity'] if pred['entity']!='C' else '-'
            score += str(min(9,int(round(pred['score'].item(),1)*10)))
        else:
            print(f"Unexpected format in prediction: {pred}")
            structure += 'X'  # Default for unrecognized labels

    outfile.write(f"{header}\n{sequence}\n{structure}\n{score}\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 protbert_runner.py <input.fasta> <output.fasta> <cpu/gpu>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    device_type = sys.argv[3]  # 'cpu' or 'gpu'

    predict_on_device(input_file, output_file, device_type)
