import torch
import yaml
import os
import os.path as osp
from model import build_model
import numpy as np
import matplotlib.pyplot as plt

FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_i3d': 2048,
    'flow_kinetics_i3d': 2048
}

# Load config
config_path = "/data1/ghufran/MiniROAD/configs/miniroad_thumos_kinetics.yaml"  # Update path if needed
checkpoint_path_rgb = "/data1/ghufran/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowFalse/ckpts/best_59.22.pth"  # Update path if needed
checkpoint_path_rgb_flow = '/data1/ghufran/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowTrue_2/ckpts/best_71.53.pth'

cfg = yaml.load(open(config_path), Loader=yaml.FullLoader)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Add missing keys to config
cfg['no_rgb'] = False
cfg['no_flow'] = True

checkpoint_path = checkpoint_path_rgb_flow if cfg['no_flow'] == False else checkpoint_path_rgb

data_path = '/data1/ghufran/MiniROAD/data/THUMOS/'
video_name = 'video_validation_0000937'

rgb_feat = data_path + 'rgb_kinetics_resnet50/' + video_name + '.npy'
flow_feat = data_path + 'flow_kinetics_bninception/' + video_name + '.npy'
target_perframe = data_path + 'target_perframe/' + video_name + '.npy'

# Load all features
rgb = torch.from_numpy(np.load(rgb_feat)).to(device)
flow = torch.from_numpy(np.load(flow_feat)).to(device)
target = torch.from_numpy(np.load(target_perframe)).to(device)

# Build model
model = build_model(cfg, device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

rgb = rgb.unsqueeze(0)
flow = flow.unsqueeze(0)

out_dict = model(rgb, flow)
pred_logit = out_dict['logits']
prob_val = pred_logit.squeeze().cpu().detach().numpy()
target_batch = target.squeeze().cpu().detach().numpy()

all_class_names = [
    "Background",
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
    "Ambiguous"
]
# Debug: Print the shape of prob_val
print(f"Shape of prob_val: {prob_val.shape}")  # Should be (N, 22)

# Calculate the number of frames where probability > 0.4 for each class
threshold_prob = 0.4  # Probability threshold
min_frame_count = 10  # Minimum number of frames required
class_indices_to_plot = []

for i, class_name in enumerate(all_class_names):
    # Count the number of frames where probability > threshold_prob
    frame_count_above_threshold = np.sum(prob_val[:, i] > threshold_prob)

    # Check if the count exceeds the minimum frame count
    if frame_count_above_threshold > min_frame_count:
        class_indices_to_plot.append(i)

# Debug: Print the classes that will be plotted
print("Classes to plot:", [all_class_names[i] for i in class_indices_to_plot])

# Plotting the probabilities for selected classes
plt.figure(figsize=(12, 8))
for i in class_indices_to_plot:
    plt.plot(prob_val[:, i], label=all_class_names[i])

plt.xlabel('Frame Number')
plt.ylabel('Probability')
plt.title(f'Predicted Probabilities for {video_name} (Classes > {threshold_prob} for > {min_frame_count} frames)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent overlap

# Save the plot
output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True)
output_path = osp.join(output_dir, f'{video_name}_probabilities_filtered.png')
plt.savefig(output_path, bbox_inches='tight')  # Ensure the legend is included in the saved plot
plt.close()

print(f"Plot saved to {output_path}")