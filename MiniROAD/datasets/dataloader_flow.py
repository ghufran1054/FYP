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
    'rgb_kinetics_resnet50_pruned': 1024,
    'rgb_features_imagenet': 2048,
    'flow_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50_self': 2048,
    'flow_kinetics_resnet50_pruned': 1024,
    'flow_kinetics_resnet50_raft': 2048,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_i3d': 2048,
    'flow_kinetics_i3d': 2048,
    'rgb_feat_finetuned': 2048,
    'flow_feat_finetuned': 2048,
    'flow_feat_farn' : 2048,
}
# ResNet50 normalization constants
mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # Mean values for R, G, B channels
std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
MEAN = [0.502]
STD = [0.502]

class THUMOSDatasetFLOW(data.Dataset):
    
    def __init__(self, cfg, mode='train', rootpath=None):
        self.root_path = rootpath
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.use_rgb = (cfg['no_rgb'] == False)
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
        self.rgb_type = cfg['rgb_type']
        self.rgb_inputs = {}

        dummy_target = np.zeros((self.window_size-1, self.num_classes))
        dummy_rgb = np.zeros((self.window_size-1, FEATURE_SIZES[self.rgb_type]))
        # if self.training:
        #     self.vids = self.vids[:1]
        # else:
        #     self.vids = [self.vids[4]]
        
        for vid in self.vids:
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            if self.use_rgb:
                rgb = np.load(osp.join('/data1/ghufran/THUMOS/', self.rgb_type, vid + '.npy'))

            # concatting dummy target at the front 
            if self.training:
                self.target_all[vid] = np.concatenate((dummy_target, target), axis=0)
                if self.use_rgb:
                    self.rgb_inputs[vid] = np.concatenate((dummy_rgb, rgb), axis=0)
                # print(f"Loading target for video {vid} with shape {target.shape}")
            else:
                self.target_all[vid] = target
                if self.use_rgb:
                    self.rgb_inputs[vid] = rgb

    
    def _get_frame_paths(self, vid):
        # There are two types of frames x and y for flow
        frame_dir = osp.join(self.root_path, vid)

        # filter those files which contain x in their name
        frame_files_x = sorted([f for f in os.listdir(frame_dir) if 'x' in f])
        frame_files_y = sorted([f for f in os.listdir(frame_dir) if 'y' in f])
        def keep_5_skip_1_grouped(files):
            grouped = []
            for i in range(0, len(files), 6):
                chunk = files[i:i+5]  # Take 5 frames
                chunk = [osp.join(frame_dir, f) for f in chunk]  # Add full path
                if len(chunk) == 5:   # Only include full 5-frame groups
                    grouped.append(tuple(chunk))
            return grouped

        # Apply it
        frame_files_x = keep_5_skip_1_grouped(frame_files_x)
        frame_files_y = keep_5_skip_1_grouped(frame_files_y)

        # If its training add path for a dummy frame at the beginning window_size - 1 times
        if self.training:
            dummy_frame_path = osp.join(frame_dir, 'dummy_frame.jpg')
            frame_files_x = [(dummy_frame_path, ) * 5] * (self.window_size - 1) + frame_files_x
            frame_files_y = [(dummy_frame_path, ) * 5] * (self.window_size - 1) + frame_files_y

        return frame_files_x, frame_files_y
    
    def _init_inputs(self):
        del self.inputs
        gc.collect()
        self.inputs = []
        self.frame_counts = {}  # Store frame counts per video
        
        # First pass: count frames for each video (without loading them)
        for vid in self.vids:
            frame_paths_x, frame_paths_y = self._get_frame_paths(vid)
            self.frame_counts[vid] = len(frame_paths_x)
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
        
        # frame_paths = self._get_frame_paths(vid)[start:end]
        frame_paths_x, frame_paths_y = self._get_frame_paths(vid)
        # Select the frames for the current window
        frame_paths_x = frame_paths_x[start:end]
        frame_paths_y = frame_paths_y[start:end]
        # Load and transform each frame
        # print(f"Loading frames for video {vid} from {start} to {end} (total frames: {len(frame_paths)})")
        frames = []

        def load_and_prepare_frames(frame_paths_x, frame_paths_y):
            all_tensors = []

            for paths_x, paths_y in zip(frame_paths_x, frame_paths_y):
                frames = []

                for px, py in zip(paths_x, paths_y):
                    # Load or create dummy x image
                    if 'dummy' in px:
                        img_x = Image.new('L', (224, 224), 0)
                    else:
                        img_x = Image.open(px).convert('L')

                    # Load or create dummy y image
                    if 'dummy' in py:
                        img_y = Image.new('L', (224, 224), 0)
                    else:
                        img_y = Image.open(py).convert('L')

                    # Apply transforms
                    img_x = self.transform(img_x)
                    img_y = self.transform(img_y)

                    # Interleave x, y
                    frames.extend([img_x, img_y])

                # Stack to shape (10, H, W)
                tensor = torch.cat(frames, dim=0)  # Shape: [10, C, H, W]
                tensor = tensor.unsqueeze(0)
                all_tensors.append(tensor)

            return all_tensors  # list of (10, H, W) tensors

        frames = load_and_prepare_frames(frame_paths_x, frame_paths_y)

        flow_input = torch.cat(frames, dim=0)  # Shape: [T, C, H, W]
        # print('Flow input shape: ', flow_input.shape)

        
        target = torch.tensor(target.astype(np.float32))
        if self.use_rgb:
            # Load flow data
            rgb_input = self.rgb_inputs[vid][start:end]
            rgb_input = torch.tensor(rgb_input.astype(np.float32))
        else:
            # Create dummy flow single number tensor to conserve memory
            rgb_input = torch.tensor(0, dtype=torch.float32)

        # print("Target shape:", target.shape)
        # print("RGB input shape:", rgb_input.shape)
        return rgb_input, flow_input, target

    def __len__(self):
        return len(self.inputs)