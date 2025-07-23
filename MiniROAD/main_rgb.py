# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine import dump, list_from_file, load
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

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

from model.rnn.rnn_rgb import MROADRGB
from torch import nn

from torch.utils.data import DataLoader


from datasets.dataloader_rgb import THUMOSDatasetRGB
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
        with torch.no_grad():
            pred_scores, gt_targets = [], []
            start = time.time()
            for rgb_input, flow_input, target in tqdm(dataloader, desc='Evaluation:', leave=False):
                rgb_input, flow_input, target = rgb_input.to(device), flow_input.to(device), target.to(device)
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
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 feature extraction')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output_prefix', type=str, help='output prefix')
    parser.add_argument(
        '--video-list', type=str, default=None, help='video file list')
    parser.add_argument(
        '--video-root', type=str, default=None, help='video root directory')
    parser.add_argument(
        '--spatial-type',
        type=str,
        default='avg',
        choices=['avg', 'max', 'keep'],
        help='Pooling type in spatial dimension')
    parser.add_argument(
        '--temporal-type',
        type=str,
        default='keep',
        choices=['avg', 'max', 'keep'],
        help='Pooling type in temporal dimension')
    parser.add_argument(
        '--long-video-mode',
        action='store_true',
        help='Perform long video inference to get a feature list from a video')
    parser.add_argument(
        '--clip-interval',
        type=int,
        default=None,
        help='Clip interval for Clip interval of adjacent center of sampled '
        'clips, used for long video inference')
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=None,
        help='Temporal interval of adjacent sampled frames, used for long '
        'video long video inference')
    parser.add_argument(
        '--multi-view',
        action='store_true',
        help='Perform multi view inference')
    parser.add_argument(
        '--dump-score',
        action='store_true',
        help='Dump predict scores rather than features')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--config_mroad', type=str, default='./configs/miniroad_thumos_kinetics.yaml')
    parser.add_argument('--eval', type=str, default=None)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--no_rgb', action='store_true')
    parser.add_argument('--no_flow', action='store_true')
    parser.add_argument('--load_GRU', type=str, default=None)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args




def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    # -------------------- Feature Head --------------------
    if not args.dump_score:
        backbone_type2name = dict(
            ResNet3dSlowFast='slowfast',
            MobileNetV2TSM='tsm',
            ResNetTSM='tsm',
        )

        if cfg.model.type == 'RecognizerGCN':
            backbone_name = 'gcn'
        else:
            backbone_name = backbone_type2name.get(cfg.model.backbone.type)
        num_segments = None
        if backbone_name == 'tsm':
            for idx, transform in enumerate(test_pipeline):
                if transform.type == 'UntrimmedSampleFrames':
                    clip_len = transform['clip_len']
                    continue
                elif transform.type == 'SampleFrames':
                    clip_len = transform['num_clips']
            num_segments = cfg.model.backbone.get('num_segments', 8)
            assert num_segments == clip_len, \
                f'num_segments and clip length must same for TSM, but got ' \
                f'num_segments {num_segments} clip_len {clip_len}'
            if cfg.model.test_cfg is not None:
                max_testing_views = cfg.model.test_cfg.get(
                    'max_testing_views', num_segments)
                assert max_testing_views % num_segments == 0, \
                    'tsm needs to infer with batchsize of multiple ' \
                    'of num_segments.'

        spatial_type = None if args.spatial_type == 'keep' else \
            args.spatial_type
        temporal_type = None if args.temporal_type == 'keep' else \
            args.temporal_type
        feature_head = dict(
            type='FeatureHead',
            spatial_type=spatial_type,
            temporal_type=temporal_type,
            backbone_name=backbone_name,
            num_segments=num_segments)
        cfg.model.cls_head = feature_head

    # ---------------------- multiple view ----------------------
    if not args.multi_view:
        # average features among multiple views
        cfg.model.cls_head['average_clips'] = 'score'
        if cfg.model.type == 'Recognizer3D':
            for idx, transform in enumerate(test_pipeline):
                if transform.type == 'SampleFrames':
                    test_pipeline[idx]['num_clips'] = 1
        for idx, transform in enumerate(test_pipeline):
            if transform.type == 'SampleFrames':
                test_pipeline[idx]['twice_sample'] = False
            # if transform.type in ['ThreeCrop', 'TenCrop']:
            if transform.type == 'TenCrop':
                test_pipeline[idx].type = 'CenterCrop'

    # -------------------- pipeline settings  --------------------
    # assign video list and video root
    if args.video_list is not None:
        cfg.test_dataloader.dataset.ann_file = args.video_list
    if args.video_root is not None:
        if cfg.test_dataloader.dataset.type == 'VideoDataset':
            cfg.test_dataloader.dataset.data_prefix = dict(
                video=args.video_root)
        elif cfg.test_dataloader.dataset.type == 'RawframeDataset':
            cfg.test_dataloader.dataset.data_prefix = dict(img=args.video_root)
    args.video_list = cfg.test_dataloader.dataset.ann_file
    args.video_root = cfg.test_dataloader.dataset.data_prefix
    # use UntrimmedSampleFrames for long video inference
    if args.long_video_mode:
        # preserve features of multiple clips
        cfg.model.cls_head['average_clips'] = None
        cfg.test_dataloader.batch_size = 1
        is_recognizer2d = (cfg.model.type == 'Recognizer2D')

        frame_interval = args.frame_interval
        for idx, transform in enumerate(test_pipeline):
            if transform.type == 'UntrimmedSampleFrames':
                clip_len = transform['clip_len']
                continue
            # replace SampleFrame by UntrimmedSampleFrames
            elif transform.type in ['SampleFrames', 'UniformSample']:
                assert args.clip_interval is not None, \
                    'please specify clip interval for long video inference'
                if is_recognizer2d:
                    # clip_len of UntrimmedSampleFrames is same as
                    # num_clips for 2D Recognizer.
                    clip_len = transform['num_clips']
                else:
                    clip_len = transform['clip_len']
                    if frame_interval is None:
                        # take frame_interval of SampleFrames as default
                        frame_interval = transform.get('frame_interval')
                assert frame_interval is not None, \
                    'please specify frame interval for long video ' \
                    'inference when use UniformSample or 2D Recognizer'

                sample_cfgs = dict(
                    type='UntrimmedSampleFrames',
                    clip_len=clip_len,
                    clip_interval=args.clip_interval,
                    frame_interval=frame_interval)
                test_pipeline[idx] = sample_cfgs
                continue
        # flow input will stack all frames
        if cfg.test_dataloader.dataset.get('modality') == 'Flow':
            clip_len = 1

        if is_recognizer2d:
            from mmaction.models import ActionDataPreprocessor
            from mmaction.registry import MODELS

            @MODELS.register_module()
            class LongVideoDataPreprocessor(ActionDataPreprocessor):
                """DataPreprocessor for 2D recognizer to infer on long video.

                Which would stack the num_clips to batch dimension, to preserve
                feature of each clip (no average among clips)
                """

                def __init__(self, num_frames=8, **kwargs) -> None:
                    super().__init__(**kwargs)
                    self.num_frames = num_frames

                def preprocess(self, inputs, data_samples, training=False):
                    batch_inputs, data_samples = super().preprocess(
                        inputs, data_samples, training)
                    # [N*M, T, C, H, W]
                    nclip_batch_inputs = batch_inputs.view(
                        (-1, self.num_frames) + batch_inputs.shape[2:])
                    # data_samples = data_samples * \
                    #     nclip_batch_inputs.shape[0]
                    return nclip_batch_inputs, data_samples

            preprocessor_cfg = cfg.model.data_preprocessor
            preprocessor_cfg.type = 'LongVideoDataPreprocessor'
            preprocessor_cfg['num_frames'] = clip_len
            print("Using Clip length: ", clip_len)

    # -------------------- Dump predictions --------------------
    args.dump = osp.join(args.output_prefix, 'total_feats.pkl')
    dump_metric = dict(type='DumpResults', out_file_path=args.dump)
    cfg.test_evaluator = [dump_metric]
    cfg.work_dir = osp.join(args.output_prefix, 'work_dir')

    return cfg




def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher

    cfg.load_from = args.checkpoint

    print(cfg)

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model_feat_ext = runner.model
    model_feat_ext.train()
    model_feat_ext = model_feat_ext.cuda()

    # Freezing some layers of the feature extractor if needed
    # Example: Freeze conv1, bn1, layer1, layer2
    for name, module in model_feat_ext.named_children():
        if name in ['conv1', 'bn1', 'layer1', 'layer2']:
            for param in module.parameters():
                param.requires_grad = False

    # # Freezing all layers of the feature extractor debugging
    # for param in model_feat_ext.parameters():
    #     param.requires_grad = False


    # combine argparse and yaml
    opt = yaml.load(open(args.config_mroad), Loader=yaml.FullLoader)
    opt.update(vars(args))
    cfg_mroad = opt


    # Create two dataloaders for test and train
    testloader = THUMOSDatasetRGB(cfg_mroad, mode='test', rootpath='/data1/ghufran/test_frames')
    print('testloader', len(testloader))
    trainloader = THUMOSDatasetRGB(cfg_mroad, mode='train', rootpath='/data1/ghufran/validation_frames')
    print('trainloader', len(trainloader))

    testloader = DataLoader(
        testloader,
        batch_size=cfg_mroad['test_batch_size'],
        shuffle=False,
        num_workers=4
    )
    trainloader = DataLoader(
        trainloader,
        batch_size=cfg_mroad['batch_size'],
        shuffle=True,
        num_workers=4
    )

    set_seed(20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    identifier = f'{cfg_mroad["model"]}_{cfg_mroad["data_name"]}_{cfg_mroad["feature_pretrained"]}_flow{not cfg_mroad["no_flow"]}'
    result_path = create_outdir(osp.join(cfg_mroad['output_path'], identifier))
    logger = get_logger(result_path)
    logger.info(cfg_mroad)

    model = MROADRGB(cfg_mroad, feature_extractor=model_feat_ext).to(device)
    evaluate = Evaluate(cfg_mroad)
    if args.load_GRU is not None:
        print('Loading GRU model from:', args.load_GRU)
        model.load_state_dict(torch.load(args.load_GRU), strict=False)

    if args.eval != None:
        model.load_state_dict(torch.load(args.eval), strict=False)
        model.eval()
        with torch.no_grad():
            mAP = evaluate(model, testloader, logger)
        logger.info(f'{cfg_mroad["task"]} result: {mAP*100:.2f} m{cfg_mroad["metric"]}')
        exit()
    
    feat_ext_params = set(p for p in model_feat_ext.parameters())

    # Filter out the overlapping parameters

    criterion = build_criterion(cfg_mroad, device)
    # train_one_epoch = build_trainer(cfg_mroad)
    feat_ext_param_ids = set(id(p) for p in model_feat_ext.parameters())
    main_params = [p for p in model.parameters() if id(p) not in feat_ext_param_ids]

    total = sum(p.numel() for p in model.parameters())
    feat_ext = sum(p.numel() for p in model_feat_ext.parameters())
    main = sum(p.numel() for p in main_params)

    print(f"Total: {total}, Main: {main}, Feat Ext: {feat_ext}, Sum: {main + feat_ext}")
    print(f"trainable parameters: {sum(p.numel() for p in main_params)}")

    from torch.optim.lr_scheduler import StepLR

    optim = torch.optim.AdamW if cfg_mroad['optimizer'] == 'AdamW' else torch.optim.Adam
    optimizer = optim(
        [{'params': main_params, 'lr': cfg_mroad['lr'] / 4, 'initial_lr' : cfg_mroad['lr'] / 4},
        {'params': model_feat_ext.parameters(), 'lr': cfg_mroad['lr'] * 0.1 / 4, 'initial_lr': cfg_mroad['lr'] * 0.1 / 4}
        ], weight_decay=cfg_mroad["weight_decay"])

    scheduler = build_lr_scheduler(cfg_mroad, optimizer, len(trainloader)) if args.lr_scheduler else None
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    writer = SummaryWriter(osp.join(result_path, 'runs')) if args.tensorboard else None
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f'Dataset: {cfg_mroad["data_name"]},  Model: {cfg_mroad["model"]}')    
    logger.info(f'lr:{cfg_mroad["lr"]} | Weight Decay:{cfg_mroad["weight_decay"]} | Window Size:{cfg_mroad["window_size"]} | Batch Size:{cfg_mroad["batch_size"]}') 
    logger.info(f'Total epoch:{cfg_mroad["num_epoch"]} | Total Params:{total_params/1e6:.1f} M | Optimizer: {cfg_mroad["optimizer"]}')
    logger.info(f'Output Path:{result_path}')

    best_mAP, best_epoch = 0, 0
    for epoch in range(1, cfg_mroad['num_epoch']+1):
        epoch_loss = train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer, scheduler=scheduler)
        trainloader.dataset._init_inputs()
        scheduler.step() if scheduler else None
        
        mAP = evaluate(model, testloader, logger)
        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = epoch
            torch.save(model.state_dict(), osp.join(result_path, 'ckpts', 'best.pth'))
        logger.info(f'Epoch {epoch} mAP: {mAP*100:.2f} | Best mAP: {best_mAP*100:.2f} at epoch {best_epoch}, iter {epoch*cfg_mroad["batch_size"]*len(trainloader)} | train_loss: {epoch_loss/len(trainloader):.4f}, lr: {optimizer.param_groups[0]["lr"]:.7f}')
        
    os.rename(osp.join(result_path, 'ckpts', 'best.pth'), osp.join(result_path, 'ckpts', f'best_{best_mAP*100:.2f}.pth'))


if __name__ == '__main__':
    main()
