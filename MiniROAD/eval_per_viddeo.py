import torch
import torch.nn as nn
from tqdm import tqdm
import time
from utils import thumos_postprocessing
from utils import *
import json
from trainer.eval_builder import EVAL
from utils import thumos_postprocessing, perframe_average_precision

from model import build_model


# model: 'MiniROAD'
# data_name: 'THUMOS'
# task: 'OAD'
# loss: 'NONUNIFORM'
# metric: 'AP'
# optimizer: 'AdamW'
# device: 'cuda'
# feature_pretrained: 'kinetics'
# root_path: './data/THUMOS'
# rgb_type: 'rgb_kinetics_resnet50'
# flow_type: 'flow_kinetics_bninception'
# annotation_type: 'target_perframe'
# video_list_path: 'data_info/video_list.json'
# output_path: './output'
# window_size: 128
# batch_size: 16 
# test_batch_size: 1 
# num_epoch: 10 
# lr: 0.0001 
# weight_decay: 0.05
# num_workers: 4
# dropout: 0.20
# num_classes: 22 # including background
# embedding_dim: 2048
# hidden_dim: 1024
# num_layers: 1
# stride: 4
# convert this above config to a python dictionary
config = {
    'model': 'MiniROAD',
    'data_name': 'THUMOS',
    'task': 'OAD',
    'loss': 'NONUNIFORM',
    'metric': 'AP',
    'optimizer': 'AdamW',
    'device': 'cuda',
    'feature_pretrained': 'kinetics',
    'root_path': './data/THUMOS',
    'rgb_type': 'rgb_kinetics_resnet50',
    'flow_type': 'flow_kinetics_bninception',
    'annotation_type': 'target_perframe',
    'video_list_path': 'data_info/video_list.json',
    'output_path': './output',
    'window_size': 128,
    'batch_size': 16,
    'test_batch_size': 1,
    'num_epoch': 10,
    'lr': 0.0001,
    'weight_decay': 0.05,
    'num_workers': 4,
    'dropout': 0.20,
    'num_classes': 22,  # including background
    'embedding_dim': 2048,
    'hidden_dim': 1024,
    'num_layers': 1,
    'stride': 4,
    'no_flow': False,
    'no_rgb': False,
}
from datasets import build_data_loader

def video_accuracy(pred_scores, gt_targets, video_names):
    video_results = {}
    for vid_name in set(video_names):
        # Get indices for current video
        vid_indices = [i for i, name in enumerate(video_names) if name == vid_name]
        # Get predictions and ground truth for current video
        vid_preds = pred_scores[vid_indices]
        vid_gts = gt_targets[vid_indices]
        
        # Filter out background frames (where gt == 0)
        non_bg_mask = (vid_gts != 0)
        vid_preds_filtered = vid_preds[non_bg_mask]
        vid_gts_filtered = vid_gts[non_bg_mask]
        
        # Skip if no non-background frames in this video
        if len(vid_gts_filtered) == 0:
            video_results[vid_name] = None  # or np.nan, or skip entirely
            continue
        
        # Calculate accuracy on non-background frames only
        vid_pred_classes = np.argmax(vid_preds_filtered, axis=1)
        correct = np.sum(vid_pred_classes == vid_gts_filtered)
        accuracy = correct / len(vid_gts_filtered)
        video_results[vid_name] = accuracy
    return video_results

class Evaluate(nn.Module):
    
    def __init__(self, cfg):
        super(Evaluate, self).__init__()
        self.data_processing = thumos_postprocessing if 'THUMOS' in cfg['data_name'] else None
        self.metric = cfg['metric']
        self.eval_method = perframe_average_precision
        self.all_class_names = json.load(open(cfg['video_list_path']))[cfg["data_name"].split('_')[0]]['class_index']
        print(self.all_class_names)
    def eval(self, model, dataloader):
        device = "cuda:0"
        model.eval()
        
        with open('analysis.txt', 'w') as results_file, torch.no_grad():
            results_file.write("Video\tAccuracy\n")  # Header

            for rgb_input, flow_input, target, vid_name in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                rgb_input = rgb_input.to(device)      # [1, C, T, H, W]
                flow_input = flow_input.to(device)    # [1, C, T, H, W]
                target = target.to(device)            # [1, T, C]

                # Forward pass
                out_dict = model(rgb_input, flow_input)
                pred_logit = out_dict['logits']       # [1, T, C] or [1, C]

                # Remove batch dim
                pred_logit = pred_logit.squeeze(0)    # [T, C] or [C]
                target = target.squeeze(0)            # [T, C] or [C]

                # Ensure both are 2D
                if pred_logit.ndim == 1:
                    pred_logit = pred_logit[np.newaxis, :]  # [1, C]
                if target.ndim == 1:
                    target = target[np.newaxis, :]          # [1, C]

                # Convert to NumPy
                prob_val = pred_logit.cpu().numpy()         # [T, C]
                target_np = target.cpu().numpy()            # [T, C]

                # Convert target from one-hot to class index
                target_indices = np.argmax(target_np, axis=1)  # [T]

                # Filter out background frames (class 0)
                non_bg_mask = target_indices != 0
                vid_preds_filtered = prob_val[non_bg_mask]         # [N, C]
                vid_gts_filtered = target_indices[non_bg_mask]      # [N]

                # Compute accuracy
                if len(vid_gts_filtered) > 0:
                    pred_classes = np.argmax(vid_preds_filtered, axis=1)
                    accuracy = np.mean(pred_classes == vid_gts_filtered)
                else:
                    accuracy = float('nan')  # or 0.0 if you prefer

                # Save results
                # Find all the unique class names in this video
                unique_classes = np.unique(target_indices)
                # Convert to class names
                class_names = [self.all_class_names[clas] for clas in unique_classes]
                # Write to file


                results_file.write(f"{vid_name[0]}\t{accuracy:.4f}\n")
                for class_name in class_names:
                    results_file.write(f"{class_name} ")
                results_file.write("\n")

        print("Video-wise accuracy saved to analysis.txt")

    def forward(self, model, dataloader):

        return self.eval(model, dataloader)


# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, default='./configs/miniroad_thumos_kinetics.yaml')
# parser.add_argument('--eval', type=str, default=None)
# parser.add_argument('--amp', action='store_true')
# parser.add_argument('--tensorboard', action='store_true')
# parser.add_argument('--lr_scheduler', action='store_true')
# parser.add_argument('--no_rgb', action='store_true')
# parser.add_argument('--no_flow', action='store_true')
# args = parser.parse_args()
evaluate = Evaluate(config)

testloader = build_data_loader(config, mode='test')
print('testloader', len(testloader))

model = build_model(config, 'cuda')
model.load_state_dict(torch.load('/data1/ghufran/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowTrue_2/ckpts/best_71.53.pth'))
evaluate(model, testloader)