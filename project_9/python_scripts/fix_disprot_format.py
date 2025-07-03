#!/usr/bin/env python3

import sys

def fix_disprot_file(input_file, output_file):
    """Fix escaped characters in DisProt TSV file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        raw = f.read()
    
    # Fix escaped characters
    fixed = raw.replace('\\n', '\n').replace('\\"', '"').replace('""', '"').replace('\\t', '\t')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fixed)
    
    print(f"Fixed file saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_disprot.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    fix_disprot_file(input_file, output_file)
