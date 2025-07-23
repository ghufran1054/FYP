import torch
import torch.utils.data as data
import numpy as np
import json
import os.path as osp
import os
from PIL import Image
import gc
from torchvision import transforms
FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'rgb_kinetics_resnet50_self': 2048,
    'rgb_features_imagenet': 2048,
    'flow_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50_self': 2048,
    'flow_kinetics_resnet50_raft': 2048,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_i3d': 2048,
    'flow_kinetics_i3d': 2048
}
# ResNet50 normalization constants
mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # Mean values for R, G, B channels
std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class THUMOSDatasetRGB(data.Dataset):
    
    def __init__(self, cfg, mode='train', rootpath=None):
        self.root_path = rootpath
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.use_flow = (cfg['no_flow'] == False)
        self.stride = cfg['stride']
        data_name = cfg['data_name']
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set'] # list of video names
        self.num_classes = cfg['num_classes']
        self.inputs = []
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),  # Resize smaller side to 256
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        
        self._load_targets(cfg)
        self._init_inputs()
        
    def _load_targets(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.target_all = {}
        self.flow_type = cfg['flow_type']
        self.flow_inputs = {}

        dummy_target = np.zeros((self.window_size-1, self.num_classes))
        dummy_flow = np.zeros((self.window_size-1, FEATURE_SIZES[cfg['flow_type']]))
        
        for vid in self.vids:
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            if self.use_flow:
                flow = np.load(osp.join('/data1/ghufran/THUMOS/', self.flow_type, vid + '.npy'))

            # concatting dummy target at the front 
            if self.training:
                self.target_all[vid] = np.concatenate((dummy_target, target), axis=0)
                if self.use_flow:
                    self.flow_inputs[vid] = np.concatenate((dummy_flow, flow), axis=0)
                # print(f"Loading target for video {vid} with shape {target.shape}")
            else:
                self.target_all[vid] = target
                if self.use_flow:
                    self.flow_inputs[vid] = flow

    
    def _get_frame_paths(self, vid):
        """Get paths to frames for a video, starting from 4th frame with stride of 6"""
        frame_dir = osp.join(self.root_path, vid)
        frame_files = sorted(os.listdir(frame_dir))
        # Start from 4th frame (index 3), then every 6 frames
        frame_files = frame_files[4::6]
        # If its training add path for a dummy frame at the beginning window_size - 1 times
        if self.training:
            dummy_frame_path = osp.join(frame_dir, 'dummy_frame.jpg')
            frame_files = [dummy_frame_path] * (self.window_size - 1) + frame_files
        return [osp.join(frame_dir, f) for f in frame_files]
    
    def _init_inputs(self):
        del self.inputs
        gc.collect()
        self.inputs = []
        self.frame_counts = {}  # Store frame counts per video
        
        # First pass: count frames for each video (without loading them)
        for vid in self.vids[::-1]:
            frame_paths = self._get_frame_paths(vid)
            self.frame_counts[vid] = len(frame_paths)
            target = self.target_all[vid]

            if target.shape[0] != self.frame_counts[vid]:
                # print(f"Frame count mismatch for video {vid}: "
                #              f"expected {target.shape[0]} frames but found {self.frame_counts[vid]} frames")
                self.frame_counts[vid] = target.shape[0]
            
            # assert len(target) == self.frame_counts[vid], "Frame count mismatch for video {}: expected {} frames but found {} frames".format(
            #     vid, len(target), self.frame_counts[vid])
            if self.training:
                seed = np.random.randint(self.stride)
                for start, end in zip(range(seed, self.frame_counts[vid], self.stride), 
                                    range(seed + self.window_size, self.frame_counts[vid]+1, self.stride)):
                    self.inputs.append([
                        vid, start, end, target[start:end]
                    ])
            else:
                start = 0
                end = self.frame_counts[vid]
                self.inputs.append([
                    vid, start, end, target[start:end]
                ])

    def __getitem__(self, index):
        vid, start, end, target = self.inputs[index]
        
        # Get the frame paths for this window

        # total_frames = end - start
        # if total_frames != len(target):
        #     print(f"Frame count mismatch for video {vid}: "
        #                      f"expected {len(target)} frames but found {total_frames} frames")
        #     # print(f"Old end index: {end} for video {vid}")
        #     end = min(end, start + len(target))
        #     # print(f"Adjusting end index to {end} for video {vid}")
        
        frame_paths = self._get_frame_paths(vid)[start:end]
        # Load and transform each frame
        # print(f"Loading frames for video {vid} from {start} to {end} (total frames: {len(frame_paths)})")
        frames = []
        for frame_path in frame_paths:
            if 'dummy_frame' in frame_path:
                # If it's a dummy frame, create a blank tensor
                img = Image.new('RGB', (224, 224), (0, 0, 0))
                # Convert to tensor and no transform
                img = transforms.ToTensor()(img)
            else:
                img = Image.open(frame_path).convert('RGB')
                img = self.transform(img)

            frames.append(img)
        
        # Stack frames along temporal dimension
        # if self.training:
        #     # Add window_size - 1 dummy frames before the actual frames
        #     dummy_frames = [torch.zeros(frames[0].shape, dtype=frames[0].dtype) for _ in range(self.window_size - 1)]
        #     frames = dummy_frames + frames

        rgb_input = torch.stack(frames, dim=0)  # Shape: [T, C, H, W]
        
        target = torch.tensor(target.astype(np.float32))
        if self.use_flow:
            # Load flow data
            flow_input = self.flow_inputs[vid][start:end]
            flow_input = torch.tensor(flow_input.astype(np.float32))
        else:
            # Create dummy flow single number tensor to conserve memory
            flow_input = torch.tensor(0, dtype=torch.float32)

        # print("Target shape:", target.shape)
        # print("RGB input shape:", rgb_input.shape)
        return rgb_input, flow_input, target

    def __len__(self):
        return len(self.inputs)