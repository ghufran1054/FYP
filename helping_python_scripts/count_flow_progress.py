import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_folder_for_flow_images(folder):
    """Check if a folder contains at least one file starting with 'flow'."""
    for file in folder.iterdir():
        if file.is_file() and file.name.startswith('flow'):
            return True
    return False

def count_folders_with_flow_images(root_folder):
    count = 0
    root_path = Path(root_folder)
    folders = [folder for folder in root_path.iterdir() if folder.is_dir()]

    # Use ThreadPoolExecutor to process folders in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = {executor.submit(check_folder_for_flow_images, folder): folder for folder in folders}
        
        for future in as_completed(futures):
            if future.result():
                count += 1
    
    return count

# Example usage
root_folder = '/data1/ghufran/mmaction2/data/kinetics400/rawframes_val'  # Replace with the path to your root folder
result = count_folders_with_flow_images(root_folder)
print(f"Number of folders containing at least one 'flow' image: {result}")