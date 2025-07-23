from src.imfe import IMFE, BatchedIMFE
from dataset.dataloader import RGBFrameFeatureDataset
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import os
import numpy as np


# # ===== STRICT CPU LIMITATION =====
# os.environ["OMP_NUM_THREADS"] = "8"         # OpenMP threads
# os.environ["MKL_NUM_THREADS"] = "8"         # Intel MKL
# os.environ["OPENBLAS_NUM_THREADS"] = "8"    # OpenBLAS
# os.environ["NUMEXPR_NUM_THREADS"] = "8"     # NumExpr
# os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # Accelerate framework on macOS
# # Now set PyTorch specific limits
# torch.set_num_threads(8)  # Main thread pool
# torch.set_num_interop_threads(8)  # Inter-operation threads
# Initialize the dataset and dataloader
rgb_root = '/data1/ghufran/validation_frames'
feature_root = '/data1/ghufran/helping_python_scripts/IMFE_train'
# dataset_train = RGBFrameFeatureDataset(rgb_root, feature_root)
rgb_root_val = '/data1/ghufran/test_frames'
feature_root_val = '/data1/ /THUMOS/flow_kinetics_bninception'
dataset_val = RGBFrameFeatureDataset(rgb_root_val, feature_root_val)

# print(f"Training dataset size: {len(dataset_train)}")
print(f"Validation dataset size: {len(dataset_val)}")



device = 'cuda' if torch.cuda.is_available() else 'cpu'
batched_model = BatchedIMFE().to(device)
batched_model.load_state_dict(torch.load('/data1/ghufran/IMFE/best_imfe.pth', map_location=device))


# dataloader = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
valloader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) if dataset_val else None

# Iterate over the dataset it will give you 6 rgb frames, 2048 vector (ignore it) and video name
# Collect the model 2048 features for each video and at the end concatenate them and save them in video_name.npy
def extract_features(model, dataloader, output, device='cuda'):
    model.eval()
    old_video_name = None
    video_features = []
    with torch.no_grad():
        for rgb_frames, _, video_name in tqdm(dataloader):
            rgb_frames = rgb_frames.to(device)  # Move to GPU if available
            features = model(rgb_frames)  # Extract features

            if video_name[0] not in {'video_validation_0000938', 'video_test_0000740'}:
                continue

            print("Continuing with video:", video_name[0])

            # Check if we are still on the same video then accumulate features
            if old_video_name is None or video_name[0] != old_video_name:
                if old_video_name is not None:
                    # Save accumulated features for the previous video
                    feature_vect = torch.cat(video_features, dim=0).cpu().numpy()

                    # Save the features to disk
                    np.save(os.path.join(output, f"{old_video_name}.npy"), feature_vect)
                # Reset for the new video
                old_video_name = video_name[0]
                video_features = []
            # Append the features for the current video
            video_features.append(features)

        # Save the last video features
        if old_video_name is not None and video_features:
            feature_vect = torch.cat(video_features, dim=0).cpu().numpy()
            np.save(os.path.join(output, f"{old_video_name}.npy"), feature_vect)

            

# Save the features to disk in current directory by name validation_features/
if not os.path.exists('validation_features'):
    os.makedirs('validation_features')
if not os.path.exists('test_features'):
    os.makedirs('test_features')
# Extract features from the training dataset
# train_features = extract_features(batched_model, dataloader, 'validation_features', device=device)
val_features = extract_features(batched_model, valloader, 'test_features', device=device)
print("Feature extraction completed and saved to disk.")