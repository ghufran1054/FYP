import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RGBFrameFeatureDataset(Dataset):
    def __init__(self, rgb_root, feature_root, transform=None, frame_size=(224, 224)):
        self.rgb_root = rgb_root
        self.feature_root = feature_root
        self.transform = transform or transforms.Compose([
            transforms.CenterCrop(224),         # resize all frames to 224x224
            transforms.ToTensor(),                 # convert to [C, H, W] float32
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to [-1, 1]
        ])

        # List all video folder names (without sorting you might get mismatches)
        self.video_names = sorted(os.listdir(feature_root))
        # Remove .npy suffix from video names
        self.video_names = [name.split('.')[0] for name in self.video_names if name.endswith('.npy')]
        self.video_names2 = sorted(os.listdir(rgb_root))

        # Take intersection of video names to ensure consistency
        self.video_names = list(set(self.video_names) & set(self.video_names2))
        self.video_names = ['video_test_0000740']
        # Preload index pointers: (video_name, start_frame_index)
        self.index = []
        for video_name in self.video_names:
            frame_dir = os.path.join(rgb_root, video_name)
            num_frames = len([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
            num_clips = num_frames // 6
            for i in range(num_clips):
                self.index.append((video_name, i * 6))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        video_name, start_idx = self.index[idx]
        frame_dir = os.path.join(self.rgb_root, video_name)

        # Load 6 frames
        frames = []
        for i in range(start_idx, start_idx + 6):
            frame_path = os.path.join(frame_dir, f"img_{i:05d}.jpg")  # adjust if frame names differ
            img = Image.open(frame_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        frames_tensor = torch.stack(frames, dim=0)  # [6, 3, H, W]

        # Load ground-truth feature
        feat_path = os.path.join(self.feature_root, video_name + ".npy")
        features = np.load(feat_path)  # shape [N//6, 2048]
        target = torch.tensor(features[start_idx // 6], dtype=torch.float32)  # [2048]

        return frames_tensor, target, video_name  # [6, 3, H, W], [2048]
