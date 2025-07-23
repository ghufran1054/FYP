import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python check_npy_shape.py <path_to_npy_file>")
    sys.exit(1)

npy_file = sys.argv[1]

try:
    data = np.load(npy_file)
    print("Shape of the array:", data.shape)
except Exception as e:
    print("Error loading .npy file:", e)
