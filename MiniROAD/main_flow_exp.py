# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
import os.path as osp
from utils import get_logger
from model import build_model
from datasets import build_data_loader
from criterions import build_criterion
from trainer import build_trainer, build_eval
from utils import *

import json
import time
from tqdm import tqdm
from utils import thumos_postprocessing, perframe_average_precision

from model.rnn.rnn_flow_exp import MROADFLOW_EXP
from torch import nn

from torch.utils.data import DataLoader

class FlowFeatureExtractor(nn.Module):
    def __init__(self, in_channels=10, out_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, H, W]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # [B, 128, H/4, W/4]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),         # [B, 256, H/8, W/8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),                                    # [B, 256, 1, 1]
        )

        self.fc = nn.Linear(256, out_dim)  # [B, 2048]

    def forward(self, x):
        # x: [B, N, C, H, W]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)        # Merge batch and temporal: [B*N, C, H, W]
        x = self.encoder(x)              # [B*N, 256, 1, 1]
        x = x.view(B * N, -1)            # [B*N, 256]
        x = self.fc(x)                   # [B*N, 2048]
        x = x.view(B, N, -1)             # [B, N, 2048]
        return x

from datasets.dataloader_flow import THUMOSDatasetFLOW
class Evaluate(nn.Module):
    
    def __init__(self, cfg):
        super(Evaluate, self).__init__()
        self.data_processing = thumos_postprocessing if 'THUMOS' in cfg['data_name'] else None
        self.metric = cfg['metric']
        self.eval_method = perframe_average_precision
        self.all_class_names = json.load(open(cfg['video_list_path']))[cfg["data_name"].split('_')[0]]['class_index']
    
    def eval(self, model, dataloader, logger):
        device = "cuda:0"
        model.eval()   
        torch.cuda.empty_cache()
        with torch.no_grad():
            pred_scores, gt_targets = [], []
            start = time.time()
            for rgb_input, flow_input, target in tqdm(dataloader, desc='Evaluation:', leave=False):
                # Not moving flow_input to device 
                rgb_input, flow_input, target = rgb_input.to(device), flow_input, target.to(device)
                # print(f'rgb_input shape: {rgb_input.shape}, flow_input shape: {flow_input.shape}, target shape: {target.shape}')
                out_dict = model(rgb_input, flow_input)
                pred_logit = out_dict['logits']
                prob_val = pred_logit.squeeze().cpu().numpy()
                target_batch = target.squeeze().cpu().numpy()
                pred_scores += list(prob_val) 
                gt_targets += list(target_batch)
            end = time.time()
            num_frames = len(gt_targets)
            result = self.eval_method(pred_scores, gt_targets, self.all_class_names, self.data_processing, self.metric)
            time_taken = end - start
            logger.info(f'Processed {num_frames} frames in {time_taken:.1f} seconds ({num_frames / time_taken :.1f} FPS)')
        
        # Print the AP of each class with class name
        for i, class_name in enumerate(self.all_class_names):
            if class_name in ["Background", "Ambiguous"]:
                continue
            logger.info(f'Class {class_name}: {result["per_class_AP"][class_name]*100:.2f}')
        return result['mean_AP']
    
    def forward(self, model, dataloader, logger):
        return self.eval(model, dataloader, logger)

def train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer=None, scheduler=None):
    epoch_loss = 0
    for it, (rgb_input, flow_input, target) in enumerate(tqdm(trainloader, desc=f'Epoch:{epoch} Training', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')):
        rgb_input, flow_input, target = rgb_input.cuda(), flow_input.cuda(), target.cuda()
        model.train()
        if scaler != None:
            with torch.cuda.amp.autocast():    
                out_dict = model(rgb_input, flow_input) 
                loss = criterion(out_dict, target)   
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_dict = model(rgb_input, flow_input) 
            loss = criterion(out_dict, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        if writer != None:
            writer.add_scalar("Train Loss", loss.item(), it+epoch*len(trainloader))
    return epoch_loss



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/miniroad_thumos_kinetics.yaml')
    parser.add_argument('--eval', type=str, default=None)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--no_rgb', action='store_true')
    parser.add_argument('--no_flow', action='store_true')
    args = parser.parse_args()

    # combine argparse and yaml
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    cfg = opt

    set_seed(20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    identifier = f'{cfg["model"]}_{cfg["data_name"]}_{cfg["feature_pretrained"]}_flow{not cfg["no_flow"]}'
    result_path = create_outdir(osp.join(cfg['output_path'], identifier))
    logger = get_logger(result_path)
    logger.info(cfg)

    # Create two dataloaders for test and train
    testloader = THUMOSDatasetFLOW(cfg, mode='test', rootpath='/data1/ghufran/test_flow')
    print('testloader', len(testloader))
    trainloader = THUMOSDatasetFLOW(cfg, mode='train', rootpath='/data1/ghufran/validation_flow')
    print('trainloader', len(trainloader))

    testloader = DataLoader(
        testloader,
        batch_size=cfg['test_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    trainloader = DataLoader(
        trainloader,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model_feat_ext = FlowFeatureExtractor().to(device)
    model = MROADFLOW_EXP(cfg, feature_extractor=model_feat_ext).to(device)
    evaluate = Evaluate(cfg)


    if args.eval != None:
        model.load_state_dict(torch.load(args.eval), strict=False)
        model.eval()
        with torch.no_grad():
            mAP = evaluate(model, testloader, logger)
        logger.info(f'{cfg["task"]} result: {mAP*100:.2f} m{cfg["metric"]}')
        exit()
    


    criterion = build_criterion(cfg, device)

    from torch.optim.lr_scheduler import StepLR

    optim = torch.optim.AdamW if cfg['optimizer'] == 'AdamW' else torch.optim.Adam
    optimizer = optim([{'params': model.parameters(), 'initial_lr': cfg['lr']}],
                        lr=cfg['lr'], weight_decay=cfg["weight_decay"])

    scheduler = build_lr_scheduler(cfg, optimizer, len(trainloader)) if args.lr_scheduler else None
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    writer = SummaryWriter(osp.join(result_path, 'runs')) if args.tensorboard else None
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f'Dataset: {cfg["data_name"]},  Model: {cfg["model"]}')    
    logger.info(f'lr:{cfg["lr"]} | Weight Decay:{cfg["weight_decay"]} | Window Size:{cfg["window_size"]} | Batch Size:{cfg["batch_size"]}') 
    logger.info(f'Total epoch:{cfg["num_epoch"]} | Total Params:{total_params/1e6:.1f} M | Optimizer: {cfg["optimizer"]}')
    logger.info(f'Output Path:{result_path}')
    # mAP = evaluate(model, testloader, logger)

    best_mAP, best_epoch = 0, 0
    for epoch in range(1, cfg['num_epoch']+1):
        epoch_loss = train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer, scheduler=scheduler)
        trainloader.dataset._init_inputs()
        scheduler.step() if scheduler else None
        
        mAP = evaluate(model, testloader, logger)
        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = epoch
            torch.save(model.state_dict(), osp.join(result_path, 'ckpts', 'best.pth'))
        logger.info(f'Epoch {epoch} mAP: {mAP*100:.2f} | Best mAP: {best_mAP*100:.2f} at epoch {best_epoch}, iter {epoch*cfg["batch_size"]*len(trainloader)} | train_loss: {epoch_loss/len(trainloader):.4f}, lr: {optimizer.param_groups[0]["lr"]:.7f}')
        
    os.rename(osp.join(result_path, 'ckpts', 'best.pth'), osp.join(result_path, 'ckpts', f'best_{best_mAP*100:.2f}.pth'))


if __name__ == '__main__':
    main()
