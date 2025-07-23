import numpy as np
import os

# Paths to the folders
folder1 = "/data1/ghufran/THUMOS/target_perframe"  # Contains .npy files with shape (N, 22)
folder2 = "/data1/ghufran/helping_python_scripts/mobile_one_rgb"  # Contains .npy files with shape (N, 2048)

# Get the list of files in both folders
files1 = sorted(os.listdir(folder1))
files2 = sorted(os.listdir(folder2))

# # Ensure both folders have the same files
# if files1 != files2:
#     print("Error: The folders do not contain the same files.")
#     exit()

# Iterate through the files and compare the N dimension
for file in files2:
    # Load the .npy files
    data1 = np.load(os.path.join(folder1, file))
    data2 = np.load(os.path.join(folder2, file))

    # Get the N dimension (first dimension)
    n1 = data1.shape[0]
    n2 = data2.shape[0]

    # Compare the N dimension
    if n1 != n2:
        print(f"Mismatch in file {file}: {n1} (target_perframe) != {n2} (rgb_kinetics_resnet50)")

        # Ask the user if they want to resize the larger file
        # user_input = input("Do you want to resize the larger file to match the smaller one? (yes/no): ").strip().lower()
        if n2 == n1 + 1:
            user_input = "yes"
        if user_input == "yes":
            # Determine which file is larger
            if n1 > n2:
                # Resize data1 to match data2's N dimension
                data1_resized = data1[:n2, :]
                np.save(os.path.join(folder1, file), data1_resized)
                print(f"Resized {file} in target_perframe to shape {data1_resized.shape}.")
            else:
                # Resize data2 to match data1's N dimension
                data2_resized = data2[:n1, :]
                np.save(os.path.join(folder2, file), data2_resized)
                print(f"Resized {file} in rgb_kinetics_resnet50 to shape {data2_resized.shape}.")
        else:
            print("No resizing performed.")
    else:
        print(f"Match in file {file}: N = {n1}")