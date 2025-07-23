import os
import numpy as np
import sys

def find_npy_files(folder):
    """Find all .npy files in the given folder."""
    return {f for f in os.listdir(folder) if f.endswith('.npy')}

def get_npy_shape(file_path):
    """Load a .npy file and return its shape along with the array."""
    try:
        data = np.load(file_path)
        return data.shape, data
    except Exception as e:
        return f"Error: {e}", None

def save_trimmed_npy(file_path, array):
    """Save the trimmed numpy array back to the same file."""
    np.save(file_path, array)
    print(f"Trimmed and saved: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_npy_shapes.py <folder1> <folder2>")
        sys.exit(1)

    folder1, folder2 = sys.argv[1], sys.argv[2]

    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        print("Invalid folder path(s)")
        sys.exit(1)

    npy_files1 = find_npy_files(folder1)
    npy_files2 = find_npy_files(folder2)

    common_files = npy_files1.intersection(npy_files2)

    if not common_files:
        print("No common .npy files found.")
        sys.exit(0)

    for file in common_files:
        path1 = os.path.join(folder1, file)
        path2 = os.path.join(folder2, file)

        shape1, data1 = get_npy_shape(path1)
        shape2, data2 = get_npy_shape(path2)

        if isinstance(shape1, str) or isinstance(shape2, str):  # Error case
            print(f"Skipping {file} due to loading error.")
            continue

        if shape1 == shape2:
            print(f"{file}: Shapes match ({shape1})")
        else:
            print(f"{file}: Shapes differ - Folder1: {shape1}, Folder2: {shape2}")

            # # Check if only the first dimension differs by 1
            # if shape1[1:] == shape2[1:] and abs(shape1[0] - shape2[0]) == 1:
            #     if shape1[0] > shape2[0]:  # Trim the larger file
            #         data1 = data1[:-1]
            #         save_trimmed_npy(path1, data1)
            #     else:
            #         data2 = data2[:-1]
            #         save_trimmed_npy(path2, data2)
            # else:
            #     print(f"Skipping {file}: Difference is not just in first dimension by 1")
