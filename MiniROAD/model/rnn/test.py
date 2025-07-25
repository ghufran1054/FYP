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
    
if __name__ == '__main__':
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
    cfg_mroad = opt

    set_seed(20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    identifier = f'{cfg_mroad["model"]}_{cfg_mroad["data_name"]}_{cfg_mroad["feature_pretrained"]}_flow{not cfg_mroad["no_flow"]}'
    result_path = create_outdir(osp.join(cfg_mroad['output_path'], identifier))
    logger = get_logger(result_path)
    logger.info(cfg_mroad)

    testloader = build_data_loader(cfg_mroad, mode='test')
    print('testloader', len(testloader))

    model = build_model(cfg_mroad, device)
    evaluate = build_eval(cfg_mroad)
    if args.eval != None:
        model.load_state_dict(torch.load(args.eval))
        mAP = evaluate(model, testloader, logger)
        logger.info(f'{cfg_mroad["task"]} result: {mAP*100:.2f} m{cfg_mroad["metric"]}')
        exit()
        
    trainloader = build_data_loader(cfg_mroad, mode='train')
    print('trainloader', len(trainloader))
    criterion = build_criterion(cfg_mroad, device)
    train_one_epoch = build_trainer(cfg_mroad)
    optim = torch.optim.AdamW if cfg_mroad['optimizer'] == 'AdamW' else torch.optim.Adam
    optimizer = optim([{'params': model.parameters(), 'initial_lr': cfg_mroad['lr']}],
                        lr=cfg_mroad['lr'], weight_decay=cfg_mroad["weight_decay"])

    scheduler = build_lr_scheduler(cfg_mroad, optimizer, len(trainloader)) if args.lr_scheduler else None
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
        trainloader.dataset._init_features()
        mAP = evaluate(model, testloader, logger)
        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = epoch
            torch.save(model.state_dict(), osp.join(result_path, 'ckpts', 'best.pth'))
        logger.info(f'Epoch {epoch} mAP: {mAP*100:.2f} | Best mAP: {best_mAP*100:.2f} at epoch {best_epoch}, iter {epoch*cfg_mroad["batch_size"]*len(trainloader)} | train_loss: {epoch_loss/len(trainloader):.4f}, lr: {optimizer.param_groups[0]["lr"]:.7f}')
        
    os.rename(osp.join(result_path, 'ckpts', 'best.pth'), osp.join(result_path, 'ckpts', f'best_{best_mAP*100:.2f}.pth'))