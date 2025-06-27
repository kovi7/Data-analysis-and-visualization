import argparse
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from features import extract_features 
import warnings
warnings.filterwarnings('ignore')

class DeepProteinNet(nn.Module):
    def __init__(self, input_dim, config):
        super(DeepProteinNet, self).__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim in config['hidden_dims']:
            layers.append(nn.Linear(current_dim, hidden_dim))

            if config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif config['activation'] == 'elu':
                layers.append(nn.ELU())
            elif config['activation'] == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif config['activation'] == 'gelu':
                layers.append(nn.GELU())

            if config['use_batch_norm']:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if config['dropout_rate'] > 0:
                layers.append(nn.Dropout(config['dropout_rate']))

            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AcidicProteinPredictor:
    def __init__(self, model_dir='results/dl_model'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_columns = None

    def load_model(self):
        try:
            # Load model config
            with open(os.path.join(self.model_dir, 'best_model_info.json')) as f:
                model_info = json.load(f)
            config = model_info['config']
            model_name = model_info['name']

            print(f"Loading model: {model_name}")
            print(f"Model accuracy: {model_info['accuracy']:.4f}")

            # Load input dim from model weights
            model_path = os.path.join(self.model_dir, f'{model_name}_model.pth')
            checkpoint = torch.load(model_path, map_location='cpu')
            input_dim = checkpoint['network.0.weight'].shape[1]

            # Load scaler and feature columns
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.model_dir, 'feature_columns.json')) as f:
                self.feature_columns = json.load(f)

            # Initialize model
            self.model = DeepProteinNet(input_dim, config)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()

            print(f"Model and preprocessing loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict_sequence(self, sequence):
        try:
            record = {
                'sequence': sequence,
                'label': 0,
                'pI': 0.0,
                'source_type': 'external'
            }
            features = extract_features(record)
            if features is None:
                return None, 0.0

            df = pd.DataFrame([features])

            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_columns]

            X_scaled = self.scaler.transform(df)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            with torch.no_grad():
                pred = self.model(X_tensor).cpu().numpy()[0][0]

            return ("acidic" if pred > 0.5 else "non-acidic", float(pred))
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0

    def predict_fasta(self, input_file, output_file):
        try:
            sequences = list(SeqIO.parse(input_file, "fasta"))
            print(f"Predicting {len(sequences)} sequences...")
            results = []

            for record in sequences:
                label, score = self.predict_sequence(str(record.seq))
                if label:
                    record.description += f" | {label} {score:.8f}"
                else:
                    record.description += " | prediction_failed 0.00000000"
                results.append(record)

            SeqIO.write(results, output_file, "fasta")
            print(f"Results written to {output_file}")

            acidic = sum(1 for r in results if "acidic" in r.description and "non-acidic" not in r.description)
            non_acidic = sum(1 for r in results if "non-acidic" in r.description)
            failed = sum(1 for r in results if "prediction_failed" in r.description)

            print("\nSummary:")
            print(f"Acidic: {acidic}")
            print(f"Non-acidic: {non_acidic}")
            print(f"Failed: {failed}")
            return True
        except Exception as e:
            print(f"FASTA processing error: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Predict acidic proteins using deep learning")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output FASTA with predictions")
    parser.add_argument("-m", "--model_dir", default="results/dl_model", help="Model directory")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    predictor = AcidicProteinPredictor(model_dir=args.model_dir)
    if not predictor.load_model():
        print("Model loading failed.")
        sys.exit(1)

    if predictor.predict_fasta(args.input, args.output):
        print("Prediction complete.")
    else:
        print("Prediction failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
