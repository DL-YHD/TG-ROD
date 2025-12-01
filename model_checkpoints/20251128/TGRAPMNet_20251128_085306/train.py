import os
import sys

import time
import datetime
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import timm.optim
import timm.scheduler

import torch
import torch.nn as nn
import yaml
from cruw import CRUW
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dataset.rod2021 import ROD2021Dataset, collate_fn, data_augment
from utils.confmap import decode_confmap
from utils.evaluate import evaluate_ols
from utils.clip_loss import clip_contrastive_loss, pad_tensor, radar_and_text_encoding_function, heteroscedastic_loss, heatmap_to_batch_class_prob, \
    PerClassCountingLoss, TextGuidedPredictor, TextVisualContrastiveLoss, TextGuidedClassWeight

from Long_CLIP.model import longclip as clip
from tools.loss_history import LossHistory


# from rodnet.core.post_processing import count_objects_by_class

# https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def train(config: dict, resume: str, device_name: str):
    
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.use_deterministic_algorithms(True, warn_only=False)

    device = torch.device(device_name)

    # Initialize model
    if config['name'] == 'mRadNet':
        from model.mRadNet import mRadNet
        model = mRadNet(
            model_cfg=config['model'],
            dataset_cfg=config['dataset']
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {config['name']}")
    
    # Initialize datasets
    print('Building training set...')
    train_set = ROD2021Dataset(
        dataset_cfg=config['dataset'],
        model_cfg=config['model'],
        data_flag='train',
        root_path=config['dataset']['root_path']
    )

    print('Building validate set...')
    valid_set = ROD2021Dataset(
        dataset_cfg=config['dataset'],
        model_cfg=config['model'],
        data_flag='valid',
        root_path=config['dataset']['root_path']
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=os.cpu_count() or 8,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count() or 8,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Initialize dataset helper
    dataset_helper = CRUW(data_root=config['dataset']['root_path'],
                          sensor_config_name=config['dataset']['helper_sensor_config'],
                          object_config_name=config['dataset']['helper_object_config'])

    num_epochs = config['train']['num_epochs']

    # Initialize optimizer
    optimizer = timm.optim.adamp.AdamP(
        model.parameters(),
        lr=config['train']['learning_rate']
    )

    # optimizer = timm.optim.adamw.AdamWLegacy(
    #     model.parameters(),
    #     lr=config['train']['learning_rate']
    # )

    # 2. 初始化CosineLRScheduler调度器：每训练n个epoch，学习率衰减为原来的0.1倍
    scheduler = timm.scheduler.cosine_lr.CosineLRScheduler(
        optimizer, num_epochs,
    )

    # 初始化调度器：在epoch 10 和 epoch 15 时进行衰减，衰减因子为0.1
    # scheduler = timm.scheduler.MultiStepLRScheduler(optimizer, decay_t=[30, 40, 45], decay_rate=0.5)

    loss_fn = nn.SmoothL1Loss().to(device)

    # 0. 初始化损失函数 (可以为不同类别设置权重，例如汽车更重要)
    # loss_count_fn = PerClassCountingLoss(weight=[1.0, 1.0, 1.5])
    # 1. 特征融合
    fusion_model = TextGuidedPredictor().to(device)
    # 2. 对比损失
    contrast_loss_fn = TextVisualContrastiveLoss().to(device)
    # 3. 类别权重调整
    weight_model = TextGuidedClassWeight().to(device)

    # Resume training
    if resume:
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Create experiment directory
    dt = datetime.datetime.now()
    exp_name = f"{config['name']}_{dt.year}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"
    print(exp_name)
    exp_dir = os.path.join(config['exp_dir'],
                           f'{dt.year}{dt.month:02d}{dt.day:02d}',
                           exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    # ================================================ #
    train_log_name = os.path.join(exp_dir, "train.log")
    with open(train_log_name, 'w'):
        pass
    # ================================================ #

    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    shutil.copyfile(f"model/{config['name']}.py",
                    os.path.join(exp_dir, 'model.py'))
    shutil.copyfile(f"train.py", os.path.join(exp_dir, 'train.py'))

    # Initialize Tensorboard
    writer = SummaryWriter(log_dir=exp_dir)
    loss_history = LossHistory(exp_dir, config['name']) # New added

    # ============================================================================================================= # 
    # 假设已得到预测结果pred和不确定度uncertainty（形状均为[B,T,H,W,C]）
    epoch_size = len(train_set) // config['train']['batch_size']

    loss_weights = config['model']['loss_weights']

    use_extra_information = config['dataset']['use_extra_information']

    use_heter_loss = config['model']['cls_uncertainty']

    chirp_id = random.randint(0, len(config['dataset']['chirps']) - 1) if config['dataset']['is_random_chirp'] else 0

    clip_model, clip_preprocess = clip.load(config['model']['clip_model']['model_path'], device=device) # New added
    clip_model.eval() # 
    start_time = time.time()
    # ============================================================================================================= # 
    for epoch in range(start_epoch, num_epochs):
        # '''
        total_loss = 0
        bar = tqdm(desc=f"Train {epoch+1}/{num_epochs}",
                   total=len(train_loader), dynamic_ncols=True)
        scheduler.step(epoch)  # timm.scheduler
        
        # Training
        model.train()

        for i, data in enumerate(train_loader):
            inputs = data['rad'].to(device)  # [B, T, R, A, C=2*chirps]
            confmap = data['confmap'].to(device)  # [B, T, R, A, classes], [4, 16, 128, 128, 3]

            # Data augmentation
            inputs, confmap = data_augment(inputs, confmap, rate=0.5)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs) # New modify
            source_loss = loss_fn(outputs['output'], confmap) 

            # =================================================================================================================================== #

            if use_extra_information:
                text_describtion = data['text_describtion'].to(device) # [4, 16, 4, 248] New added

                text_encode = clip_model.encode_text(text_describtion[:,:,chirp_id,:].view(-1, config['model']['clip_model']['vector_length']))
                
                init_fused_preds_r, init_fused_preds_a, visual_preds, text_guide = fusion_model(outputs['output'], text_encode)
                
                fused_preds_a = init_fused_preds_a + visual_preds
                fused_preds_r = init_fused_preds_r + visual_preds
                # fused_preds_ra = init_fused_preds_a + init_fused_preds_r

                weighted_preds, class_weights = weight_model(outputs['output'], text_encode)
                confmap_prob = heatmap_to_batch_class_prob(confmap)

                if use_heter_loss:
                    heter_loss = heteroscedastic_loss(outputs['class_mean'], outputs['class_log_var'], confmap)
                    contrast_loss = contrast_loss_fn(outputs['output'], text_encode)
                else:
                    heter_loss = 0
                    contrast_loss = 0

                # uncertainty_map = torch.exp(0.5 * outputs['class_log_var']) # 等价于开根号

                # # 计算权重：不确定度越低，权重越高（取所有类别的平均不确定度）
                # uncertainty_map = 1.0 / (uncertainty_map + 1e-8)  # 不确定度低 → 权重高
                # uncertainty_map = uncertainty_map / uncertainty_map.max()  # 归一化权重
                # uncertainty_pred = uncertainty_map * outputs['output']

                # uncertainty_pred_loss = loss_fn(uncertainty_pred, confmap)
                # 过滤：仅保留不确定度低于阈值的预测，其余标记为“未知”
                # reliable_mask = uncertainty_map < threshold  # 生成掩码[B,1,T,H,W]，True表示可靠
                # filtered_pred = outputs['output'] * reliable_mask + (-1) * (~reliable_mask)  # 不可靠区域标记为-1

                aux_loss = loss_fn(fused_preds_a, confmap) + loss_fn(fused_preds_r, confmap) + loss_fn(class_weights, confmap_prob) + loss_fn(weighted_preds, confmap) + loss_weights * (contrast_loss + heter_loss)

                loss = source_loss + aux_loss
            else:
                loss = source_loss
            # =================================================================================================================================== #
            
            loss.backward()
            total_loss += loss.item()
            
            optimizer.step()

            bar.update(1)
            writer.add_scalar('train/loss', loss.item(),
                               epoch * len(train_loader) + i)
            
            # =================================================================================================================================== #
            # learning rate
            Nowlr = optimizer.state_dict()['param_groups'][0]['lr']
            # 计算时间
            seconds = (time.time() - start_time)
            day = seconds / 86400
            hour = seconds % 86400 / 3600
            minute = seconds % 86400 % 3600 / 60
            second = seconds % 86400 % 3600 % 60
            if i % config['train']['log_step'] == 0:
                with open(train_log_name, 'a+') as f_log:
                        f_log.write(
                            'epoch %2d, iter %4d: loss: %.8f | lr: %.8f  | spend time:%2d days %2d hours %2d minutes %2d seconds\n' %
                            (epoch + 1, i + 1, loss,  Nowlr, day, hour, minute, second))
            # =================================================================================================================================== #
        loss_history.append_loss(total_loss / (epoch_size + 1)) # 
        bar.close()
        # ''' 
        
        # Validation
        bar = tqdm(desc=f"Valid {epoch+1}/{num_epochs}",
                   total=len(valid_loader), dynamic_ncols=True)
        model.eval()
        fusion_model.eval()
        weight_model.eval()

        predictions = {}  # {seq: {frame: [[R, A, class, conf], ...]}}
        for seq in valid_set.rads.keys():
            predictions[seq] = {}
        losses = []
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                assert len(data['rad']) == 1, f"Testing batch size must be 1"

                inputs = data['rad'].to(device)  # [1, T, R, A, C=2*chirps]
                confmap = data['confmap'].to(device)  # [1, T, R, A, classes]
                # =========================================================================== #

                # Forward pass
                outputs = model(inputs)

                source_loss = loss_fn(outputs['output'], confmap)

                # =================================================================================================================================== #
                
                if use_extra_information:
                    text_describtion = data['text_describtion'].to(device) # New added
                    text_encode = clip_model.encode_text(text_describtion[:,:,chirp_id,:].view(-1, config['model']['clip_model']['vector_length']))

                    init_fused_preds_r, init_fused_preds_a, visual_preds, text_guide = fusion_model(outputs['output'], text_encode)
                    fused_preds_a = init_fused_preds_a + visual_preds
                    fused_preds_r = init_fused_preds_r + visual_preds
                    # fused_preds_ra = init_fused_preds_a + init_fused_preds_r

                    weighted_preds,  class_weights = weight_model(outputs['output'], text_encode)
                    confmap_prob = heatmap_to_batch_class_prob(confmap)
                    contrast_loss = contrast_loss_fn(outputs['output'], text_encode)

                    if use_heter_loss:
                        heter_loss = heteroscedastic_loss(outputs['class_mean'], outputs['class_log_var'], confmap)
                        contrast_loss = contrast_loss_fn(outputs['output'], text_encode)
                    else:
                        heter_loss = 0
                        contrast_loss = 0
                    # uncertainty_map = torch.exp(0.5 * outputs['class_log_var'])

                    # # 计算权重：不确定度越低，权重越高（取所有类别的平均不确定度）
                    # uncertainty_map = 1.0 / (uncertainty_map + 1e-8)  # 不确定度低 → 权重高
                    # uncertainty_map = uncertainty_map / uncertainty_map.max()  # 归一化权重
                    # uncertainty_pred = uncertainty_map * outputs['output']

                    # uncertainty_pred_loss = loss_fn(uncertainty_pred, confmap)
                    # 过滤：仅保留不确定度低于阈值的预测，其余标记为“未知”
                    # reliable_mask = uncertainty_map < threshold  # 生成掩码[B,1,T,H,W]，True表示可靠
                    # filtered_pred = outputs['output'] * reliable_mask + (-1) * (~reliable_mask)  # 不可靠区域标记为-1

                    aux_loss = loss_fn(fused_preds_a, confmap) + loss_fn(fused_preds_r, confmap) + loss_fn(class_weights, confmap_prob) + loss_fn(weighted_preds, confmap) + loss_weights * (contrast_loss + heter_loss)

                    loss = source_loss + aux_loss
                else:
                    loss = source_loss
                # =================================================================================================================================== #

                losses.append(loss.item())

                bar.update(1)

                seq, frames = data['seq'][0], data['frames'][0]
                for j, frame in enumerate(frames):
                    if frame in predictions[seq]:
                        continue
                    pre_confmap = outputs['output'][0, j].cpu()
                    predictions[seq][frame] = decode_confmap(
                        pre_confmap, config['dataset'], config['model'])
        bar.close()
        avg_loss = np.mean(losses)
        writer.add_scalar('valid/loss', avg_loss, epoch)

        AP, AR = evaluate_ols(dataset_helper, predictions, config, 'valid', threshold=0.0)
        AP_total, AP_pedestrian, AP_cyclist, AP_car = AP[3], AP[0], AP[1], AP[2]
        AR_total, AR_pedestrian, AR_cyclist, AR_car = AR[3], AR[0], AR[1], AR[2]

        writer.add_scalar('valid/OLS_AP', AP_total, epoch)
        writer.add_scalar('valid/OLS_AR', AR_total, epoch)

        print(f"Valid {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.3f}") 
        print("AP_total: %.4f | AP_pedestrian: %.4f | AP_cyclist: %.4f | AP_car: %.4f" % (AP_total, AP_pedestrian, AP_cyclist, AP_car))
        print("AR_total: %.4f | AR_pedestrian: %.4f | AR_cyclist: %.4f | AR_car: %.4f" % (AR_total, AR_pedestrian, AR_cyclist, AR_car))
        
        # print(f"AP_total: {AP_total:.3f}, AR_total: {AR_total:.3f}, AP/AR: {AP_total/AR_total:.3f}")
        # print(f"AP_pedestrian: {AP_pedestrian:.3f}, AR_pedestrian: {AR_pedestrian:.3f}, AP/AR: {AP_pedestrian/AR_pedestrian:.3f}")
        # print(f"AP_cyclist: {AP_cyclist:.3f}, AR_cyclist: {AR_cyclist:.3f}, AP/AR: {AP_cyclist/AR_cyclist:.3f}")


        # ====================================================================================================================================================== #
        with open(train_log_name, 'a+') as AP_AR_log:
            AP_AR_log.write(f'validing {epoch+1}/{num_epochs} model ...\n') 
            AP_AR_log.write(f"Avg Loss: {avg_loss:.3f}\n")
            AP_AR_log.write("AP_total: %.4f | AP_pedestrian: %.4f | AP_cyclist: %.4f | AP_car: %.4f \n" % (AP_total, AP_pedestrian, AP_cyclist, AP_car))
            AP_AR_log.write("AR_total: %.4f | AR_pedestrian: %.4f | AR_cyclist: %.4f | AR_car: %.4f \n" % (AR_total, AR_pedestrian, AR_cyclist, AR_car))

            # AP_AR_log.write(f"AP_total: {AP_total:.3f}, AR_total: {AR_total:.3f}, AP/AR: {AP_total/AR_total:.3f}\n")
            # AP_AR_log.write(f"AP_pedestrian: {AP_pedestrian:.3f}, AR_pedestrian: {AR_pedestrian:.3f}, AP/AR: {AP_pedestrian/AR_pedestrian:.3f}\n")
            # AP_AR_log.write(f"AP_cyclist: {AP_cyclist:.3f}, AR_cyclist: {AR_cyclist:.3f}, AP/AR: {AP_cyclist/AR_cyclist:.3f}\n")
            # AP_AR_log.write(f"AP_car: {AP_car:.3f}, AR_car: {AR_car:.3f}, AP/AR: {AP_car/AR_car:.3f}\n")
        # ====================================================================================================================================================== #
                
        # Save the model
        if (epoch % 1 == 0 and AP_total > .86) or AR_total > .91 or (epoch == num_epochs - 1):
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(exp_dir, f'checkpoint_{epoch+1}.pt'))

            # ================================================================= #
            if use_extra_information:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': fusion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(exp_dir, f'checkpoint_fusion_model_{epoch+1}.pt'))

                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': weight_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(exp_dir, f'checkpoint_weight_model_{epoch+1}.pt'))
            
                torch.save({
                'epoch': epoch+1,
                'model_state_dict': contrast_loss_fn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(exp_dir, f'checkpoint_constrast_model_{epoch+1}.pt'))
            # ================================================================= #


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config',
                        default='./config/mRadNet.yaml', type=str)
    parser.add_argument('-r', '--resume', default='', type=str)
    parser.add_argument('-d', '--device', default='cuda:0', type=str)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config, resume=args.resume, device_name=args.device)
