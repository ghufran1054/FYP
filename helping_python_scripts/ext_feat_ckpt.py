import torch

# Input and output paths
in_path = "/data1/ghufran/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowTrue_189/ckpts/best_51.87.pth"
in_path = '/data1/ghufran/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowTrue_170/ckpts/best.pth'
out_path = "feature_extractor_rgb.pth"

# Load state_dict directly
state_dict = torch.load(in_path, map_location='cpu')

# Filter and rename keys that contain 'feature_extractor'
filtered_sd = {
    k.replace('feature_extractor.', ''): v
    for k, v in state_dict.items()
    if 'feature_extractor.' in k
}

# Save to new checkpoint
torch.save(filtered_sd, out_path)
