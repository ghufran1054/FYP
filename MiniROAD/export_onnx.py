import torch
import yaml
import os
import os.path as osp
from model import build_model
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'rgb_kinetics_resnet50_self': 2048,
    'flow_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50_self': 2048,
    'flow_kinetics_resnet50_raft': 2048,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_i3d': 2048,
    'flow_kinetics_i3d': 2048
}

class MROAD(nn.Module):
    def __init__(self, cfg):
        super(MROAD, self).__init__()
        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        
        self.input_dim = 0
        if self.use_rgb:
            self.input_dim += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow:
            self.input_dim += FEATURE_SIZES[cfg['flow_type']]

        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        self.out_dim = cfg['num_classes']

        self.embedding_dim = cfg['embedding_dim']
        self.relu = nn.ReLU()
        
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, 
                          self.num_layers, batch_first=True)

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout'])
        )
        
        self.f_classification = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim)
        )

    def forward(self, rgb_input, flow_input, h0):
        """ Forward pass with explicit h0 as input. """

        # Concatenate RGB & Flow features if both are used
        if self.use_rgb and self.use_flow:
            x = torch.cat((rgb_input, flow_input), dim=2)
        elif self.use_rgb:
            x = rgb_input
        elif self.use_flow:
            x = flow_input
        else:
            raise ValueError("At least one of RGB or Flow input must be used.")

        x = self.layer1(x)  # Linear -> LayerNorm -> ReLU -> Dropout
        ht, h1 = self.gru(x, h0)  # GRU output
        ht = self.relu(ht)  # Activation

        logits = self.f_classification(ht)  # Classification head
        pred_scores = F.softmax(logits, dim=-1)  # Normalize scores

        return pred_scores, h1  # Return scores + updated hidden state
# Load config
config_path = "./configs/miniroad_thumos_kinetics.yaml"  # Update path if needed
# checkpoint_path = "/data1/ghufran/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowTrue_2/ckpts/best_71.53.pth"  # Update path if needed
checkpoint_path = "/data1/ghufran/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowFalse_2/ckpts/best_56.78.pth"
checkpoint_path = "/data1/ghufran/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowFalse/ckpts/best_59.22.pth"
# checkpoint_path = './output/MiniROAD_THUMOS_kinetics_flowTrue/ckpts/best_64.17.pth'
# checkpoint_path = '/home/ghufran/FYP/Lambda-Data/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowTrue_192/ckpts/best_64.65.pth'
checkpoint_path = '/home/ghufran/FYP/Lambda-Data/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowTrue_198/ckpts/best_66.30.pth'
checkpoint_path = '/home/ghufran/FYP/Lambda-Data/MiniROAD/output/MiniROAD_THUMOS_kinetics_flowFalse_10/ckpts/best.pth'

cfg = yaml.load(open(config_path), Loader=yaml.FullLoader)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Add missing keys to config
cfg['no_rgb'] = False
cfg['no_flow'] = True

batch_size = 1
seq_len = 1  # Frame-by-frame inference
rgb_feat_dim = FEATURE_SIZES[cfg['rgb_type']]
flow_feat_dim = FEATURE_SIZES[cfg['flow_type']]
hidden_dim = cfg['hidden_dim']
num_layers = cfg['num_layers']

dummy_rgb = torch.randn(batch_size, seq_len, rgb_feat_dim).to(device)
dummy_flow = torch.randn(batch_size, seq_len, flow_feat_dim).to(device)
dummy_h0 = torch.randn(num_layers, batch_size, hidden_dim).to(device)
# Build model
model = MROAD(cfg).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

onnx_path = "mroad_rgb_mob.onnx"
# Export to ONNX
torch.onnx.export(
    model,
    (dummy_rgb, dummy_flow, dummy_h0),
    onnx_path,
    input_names=["rgb_input", "flow_input", "h0"],
    output_names=["pred_scores", "h1"],
    opset_version=11
)

print(f"Model exported to {onnx_path}")
