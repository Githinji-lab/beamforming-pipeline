# Main entry point for the beamforming project

import os
import sys

# Add src to path
sys.path.append('src')

def main():
    print("Beamforming ML Pipeline")
    print("=======================")

    # Create necessary directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("Available commands:")
    print("1. Generate training data: python pipeline/data_gen.py")
    print("2. Train SAC agent: python pipeline/train.py")
    print("3. Evaluate agent: python pipeline/evaluate.py")
    print("4. Visualize results: python pipeline/visualize.py")

if __name__ == "__main__":
    main()