import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_builder import META_ARCHITECTURES
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

class MROADFLOW_EXP(nn.Module):
    
    def __init__(self, cfg, feature_extractor=None):
        super(MROADFLOW_EXP, self).__init__()
        self.use_rgb = not cfg['no_rgb']
        self.use_flow = not cfg['no_flow']

        self.input_dim = 0
        if self.use_rgb:
            self.input_dim += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow:
            self.input_dim += FEATURE_SIZES[cfg['flow_type']]

        
        self.feature_extractor = feature_extractor


        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        self.out_dim = cfg['num_classes']
        self.window_size = cfg['window_size']

        self.relu = nn.ReLU()
        self.embedding_dim = cfg['embedding_dim']
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout']),
        )
        self.f_classification = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        # self.h0 = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.h0 = torch.zeros(self.num_layers, 1, self.hidden_dim)
        self.save = 1

    def forward(self, rgb_input, flow_input):

        x = flow_input
        # x = self.feature_extractor(x)
        B = x.shape[0]
        frames = x.shape[1] * B
        x = x.view(-1, *x.shape[2:])  # Flatten to B*N, C, H, W
        x = x.unsqueeze(0)
        max_size = 1024
        if frames > max_size:
            x_chunks = torch.split(x, max_size, dim=1)
            feature_list = []
            for chunk in x_chunks:
                # print('Chunk shape: ', chunk.shape)
                Bc, Nc, Cc, Hc, Wc = chunk.shape
                chunk = chunk.to('cuda')
                # chunk = chunk.reshape(Bc * Nc, Cc, Hc, Wc)  # Flatten to B*N1
                feat = self.feature_extractor(chunk)       # Output: (B, N1, D)
                # Make the output shape (B*N1, D)
                feat = feat.view(Bc * Nc, -1)
                # print('Feature shape: ', feat.shape)
                feature_list.append(feat.cpu())

                del chunk, feat
                torch.cuda.empty_cache()
            
            x = torch.cat(feature_list, dim=0).to('cuda')  # Concatenate along batch dimension
            x = x.view(B, -1, x.shape[-1])  # Reshape back to (B, N, D)
        else:
            x = x.to('cuda')
            # print('Input shape: ', x.shape)
            x = self.feature_extractor(x)
            # Reshape to (B, N, D)
            x = x.view(B, -1, x.shape[-1])
        
        if self.use_rgb and self.use_flow:
            x = torch.cat((rgb_input, x), 2)



        x = self.layer1(x)
        B, _, _ = x.shape
        h0 = self.h0.expand(-1, B, -1).to(x.device)
        ht, _ = self.gru(x, h0) 
        ht = self.relu(ht)
        # ht = self.relu(ht + x)
        logits = self.f_classification(ht)
        out_dict = {}
        if self.training:
            out_dict['logits'] = logits
        else:
            pred_scores = F.softmax(logits, dim=-1)
            out_dict['logits'] = pred_scores
        return out_dict