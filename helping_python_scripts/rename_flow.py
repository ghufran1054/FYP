import os

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.startswith("flow_x_"):
            new_name = filename.replace("flow_x_", "x_")
        elif filename.startswith("flow_y_"):
            new_name = filename.replace("flow_y_", "y_")
        else:
            continue
        
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")


# Get all video folders inside validation_flow folder
folders = os.listdir("/data1/ghufran/test_flow")  # Change this to your folder path
print("Found folders:", len(folders))
y = input("Do you want to continue? (y/n): ")
if y.lower() != "y":
    exit()

for folder in folders:
    folder_path = os.path.join("test_flow", folder)
    rename_files(folder_path)

