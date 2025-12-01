import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from cruw import CRUW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.rod2021 import ROD2021Dataset, collate_fn
from utils.confmap import decode_confmap
from utils.evaluate import evaluate_ols

from rodnet.utils.visualization import visualize_test_img, visualize_test_img_wo_gt
from rodnet.core.radar_processing import chirp_amp
from rodnet.core.post_processing import post_process_single_frame

from Long_CLIP.model import longclip as clip
from utils.clip_loss import PerClassCountingLoss, TextGuidedPredictor, TextVisualContrastiveLoss, TextGuidedClassWeight

from PIL import Image
from tools.evison_src.Evison import Display, show_network


# https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def test(config: dict, resume: str, device_name: str):

    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    device = torch.device(device_name)

    # Initialize datasets
    if args.data_flag == 'valid':
        print('Building validation set...')
        valid_set = ROD2021Dataset(
            dataset_cfg=config['dataset'],
            model_cfg=config['model'],
            data_flag='valid',
            root_path=config['dataset']['root_path']
        )
    elif args.data_flag == 'test':
        print('Building testing set...')
        valid_set = ROD2021Dataset(
            dataset_cfg=config['dataset'],
            model_cfg=config['model'],
            data_flag='test',
            root_path=config['dataset']['root_path']
        )
    else:
        raise ValueError(f"Unknown data flag: {args.data_flag}")


    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Initialize dataset helper
    dataset_helper = CRUW(data_root=config['dataset']['root_path'],
                          sensor_config_name=config['dataset']['helper_sensor_config'],
                          object_config_name=config['dataset']['helper_object_config'])

    predictions = {}  # {seq: {frame: [[R, A, class, conf], ...]}}
    for seq in valid_set.rads.keys(): # seq = '2019_09_29_ONRD005'
        predictions[seq] = {}

    # Initialize model
    if config['name'] == 'mRadNet':
        from model.mRadNet import mRadNet
        model = mRadNet(
            model_cfg=config['model'],
            dataset_cfg=config['dataset']
        ).to(device)
    elif config['name'] == 'TGRAPMNet':
        from model.TGRAPMNet import TGRAPMNet
        model = TGRAPMNet(
            model_cfg=config['model'],
            dataset_cfg=config['dataset']
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {config['name']}")

    checkpoint = torch.load(resume, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ==================================================================== # 

    if config['name'] == 'TGRAPMNet':
        chirp_id = random.randint(0, len(config['dataset']['chirps']) - 1) if config['dataset']['is_random_chirp'] else 0
        use_extra_information = config['dataset']['use_extra_information']
    
        if use_extra_information:
            
            # 0. Initialize loss function (weights can be set for different categories, such as cars being more important)
            # loss_count_fn = PerClassCountingLoss(weight=[1.0, 1.0, 1.5])
            
            # 1. Feature Fusion
            fusion_model = TextGuidedPredictor().to(device)
            # 2. Contrastive Loss
            contrast_loss_model = TextVisualContrastiveLoss().to(device)
            # 3. Category Weight Adjustment
            weight_model = TextGuidedClassWeight().to(device)

            clip_model, clip_preprocess = clip.load(config['model']['clip_model']['model_path'], device=device) # New added
            clip_model.eval()
            fusion_checkpoint = torch.load(resume.replace('checkpoint_', 'checkpoint_fusion_model_'), map_location=device, weights_only=True)
            fusion_model.load_state_dict(fusion_checkpoint['model_state_dict']) 
            fusion_model.eval()
            
            constrast_checkpoint = torch.load(resume.replace('checkpoint_', 'checkpoint_constrast_model_'), map_location=device, weights_only=True)
            contrast_loss_model.load_state_dict(constrast_checkpoint['model_state_dict']) 
            contrast_loss_model.eval()

            weight_checkpoint = torch.load(resume.replace('checkpoint_', 'checkpoint_weight_model_'), map_location=device, weights_only=True)
            weight_model.load_state_dict(weight_checkpoint['model_state_dict']) 
            weight_model.eval()
    else:
        chirp_id = 0
    # ==================================================================== #

    bar = tqdm(total=len(valid_loader))
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            assert len(data['rad']) == 1, f"Testing batch size must be 1"

            inputs = data['rad'].to(device)  # [1, T, R, A, C=2*chirps]
            
            if args.data_flag == 'valid':
                confmap_gt = data['confmap'].to(device)  # [1, T, R, A, classes]
                confmap_gt = confmap_gt.permute(0, 4, 1, 2, 3).to('cpu').numpy() # (1, 16, 128, 128, 3) -> (1, 3, 16, 128, 128)
                
            text_describtion = data['text_describtion'].to(device) # ['text']

            # Forward pass
            outputs = model(inputs)['output']
            
            bar.update(1)

            seq, frames = data['seq'][0], data['frames'][0]

            # =============================================================================== #

            seq_res_dir = os.path.join(config['dataset']['vis_res_dir'], config['name'], args.data_flag, seq)
            if not os.path.exists(seq_res_dir):
                os.makedirs(seq_res_dir)
            seq_res_viz_dir = os.path.join(seq_res_dir, 'rod_viz')
            if not os.path.exists(seq_res_viz_dir):
                os.makedirs(seq_res_viz_dir)

            radar_input = inputs.permute(0, 2, 5, 1, 3, 4).to('cpu').numpy() # [1, 16, 4, 128, 128, 2] -> [1, 4, 2, 16, 128, 128]
            
            if config['name'] == 'TGRAPMNet' and use_extra_information:
                text_encode = clip_model.encode_text(text_describtion[:,:,chirp_id,:].view(-1, config['model']['clip_model']['vector_length']))
                init_fused_preds_r, init_fused_preds_a, visual_preds, text_guide = fusion_model(outputs, text_encode)
                
                fused_preds_r = init_fused_preds_r + visual_preds
                fused_preds_a = init_fused_preds_a + visual_preds

                weighted_preds, class_weights = weight_model(outputs, text_encode)

                # fuse_outputs = (outputs + fused_preds + weighted_preds) / 3
                # fuse_outputs = (fused_preds + weighted_preds) / 2
                # fuse_outputs = fused_preds
                # fuse_outputs = (outputs + weighted_preds) / 2 
                # fuse_outputs = fused_preds_r  
                # fuse_outputs = fused_preds_a                        
                # fuse_outputs = (outputs + fused_preds) / 2 + weighted_preds
                fuse_outputs = (outputs + (fused_preds_r + fused_preds_a) / 2 + weighted_preds) / 3

                # confmap_pred = torch.sigmoid(fuse_outputs).permute(0, 4, 1, 2, 3).to('cpu').numpy() # [1, 16, 128, 128, 3] -> [1, 3, 16, 128, 128]
                confmap_pred = fuse_outputs.permute(0, 4, 1, 2, 3).to('cpu').numpy()
            else:
                # confmap_pred = torch.sigmoid(outputs).permute(0, 4, 1, 2, 3).to('cpu').numpy() # [1, 16, 128, 128, 3] -> [1, 3, 16, 128, 128]
                confmap_pred = outputs.permute(0, 4, 1, 2, 3).to('cpu').numpy()
            # =============================================================================== #
            for j, frame in enumerate(frames):
                if frame in predictions[seq]:
                    continue
                confmap = outputs[0][j].cpu()
                predictions[seq][frame] = decode_confmap(
                                                         confmap, config['dataset'], config['model'])
                
                # =============================================================================== #
                
                fig_name = os.path.join(config['dataset']['vis_res_dir'], config['name'], args.data_flag, seq, 'rod_viz', '%010d.jpg' % (frame)) # image path
                img_name = os.path.join(config['dataset']['root_path'], 'sequences', args.data_flag, seq, 'IMAGES_0', '%010d.jpg' % (frame))

                # image = Image.open(img_name).resize((128,128))
                # feature_map, final_out = display.forward(radar_input)
                # display.save(image)
                # image.save(fig_name.replace('.jpg','_model.jpg'))

                confmap_pred_idx = confmap_pred[0, :, j, :, :]
                radar_input_amp = chirp_amp(radar_input[0, chirp_id, :, j, :, :], 'ROD2021')
                res_final = post_process_single_frame(confmap_pred_idx, dataset_helper, config)
                
                if args.data_flag == 'valid':
                    confmap_gt_idx = confmap_gt[0, :, j, :, :]
                    visualize_test_img(fig_name, img_name, radar_input_amp, confmap_pred_idx, confmap_gt_idx, res_final,
                                       config, img_save_split=args.img_save_split)
                else:
                    # There are No RGB imgs, Have some problems 
                    # visualize_test_img_wo_gt(fig_name, img_name, radar_input_amp, confmap_pred_idx, res_final,
                    #                          config, sybl=args.symbol)
                    pass
                # =============================================================================== #
    bar.close()

    if args.data_flag == 'valid':
        AP, AR = evaluate_ols(dataset_helper, predictions, config, 'valid', 0.7)
        AP_total, AP_pedestrian, AP_cyclist, AP_car = AP[3], AP[0], AP[1], AP[2]
        AR_total, AR_pedestrian, AR_cyclist, AR_car = AR[3], AR[0], AR[1], AR[2]
    
        print(f"valid - OLS AP_total: {AP_total:.4f}, AR_total: {AR_total:.4f}")
    elif args.data_flag == 'test':
        _, _ = evaluate_ols(dataset_helper, predictions, config, 'test', 0.0)
        print(f"Test Sets Predict Complete!")
    else:
        raise ValueError(f"Unknown data flag: {args.data_flag}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config',
                        default='./config/TGRAPMNet.yaml', type=str)
    parser.add_argument('-r', '--resume', default='model_checkpoints/20251102/mRadNet_20251102_181014/checkpoint_5.pt', type=str)
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument('-f', '--data_flag', default='valid', type=str)
    parser.add_argument('-i', '--img_save_split', action="store_true", help='save image as whole or split')
    parser.add_argument('-s', '--symbol', action="store_true", help='use symbol or text+score')
    parser.add_argument('-n', '--use_noise_channel', action="store_true", help="use noise channel or not")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    test(config, resume=args.resume, device_name=args.device)
