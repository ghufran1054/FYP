import os
import torch
import pickle
import sys
import numpy as np

def find_pkl_files(folder):
    """Find all .pkl files in the given folder."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl')]

def convert_and_save(pkl_file, out_folder):
    """Load a .pkl file, convert it to NumPy, and save as .npy."""
    try:
        with open(pkl_file, 'rb') as f:
            tensor = pickle.load(f)
        
        if isinstance(tensor, torch.Tensor):
            np_array = tensor.numpy()  # Convert to NumPy
            out_file = os.path.join(out_folder, os.path.basename(pkl_file).replace('.pkl', '.npy'))
            np.save(out_file, np_array)  # Save as .npy
            print(f"Saved: {out_file}")
        else:
            print(f"Skipping {pkl_file}: Not a tensor")
    except Exception as e:
        print(f"Error processing {pkl_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_pkl_to_npy.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print("Invalid input folder path")
        sys.exit(1)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    pkl_files = find_pkl_files(input_folder)

    if not pkl_files:
        print("No .pkl files found.")
        sys.exit(0)

    for pkl_file in pkl_files:
        convert_and_save(pkl_file, output_folder)
