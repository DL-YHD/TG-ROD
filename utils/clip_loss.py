
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np


# 方式1：文本特征引导的预测修正（特征融合）
class TextGuidedPredictor(nn.Module):
    def __init__(self, class_dim=3, text_dim=512, hidden_dim=128):
        super().__init__()
        # 文本特征投影层，将512维文本特征映射到适合视觉特征的维度
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, class_dim)
        )
        # 融合注意力层，学习文本与视觉特征的关联
        self.attention = nn.MultiheadAttention(
            embed_dim=class_dim,
            num_heads=1,
            batch_first=True
        )
        
    def forward(self, visual_preds, text_features):
        """
        参数:
            visual_preds: 视觉模型预测结果，形状[B, T, H, W, C] 
                          (B=4, T=16, H=128, W=128, C=3)
            text_features: CLIP编码的文本特征，形状[B, 512]
        返回:
            fused_preds: 融合后的预测结果，形状不变
        """
        B, T, R, A, C = visual_preds.shape
        text_features = text_features.to(torch.float32)

        # 文本特征投影并广播到所有帧和空间位置
        text_guide = self.text_proj(text_features) # [B, C]
        text_guide = text_guide.view(B, T, -1) # [B, T, C]
        text_guide = text_guide.unsqueeze(2).unsqueeze(3)  # [B, T, 1, 1, C]
        text_guide = text_guide.expand(-1, -1, R, A, -1)  # [B, T, H, W, C]
        
        # 注意力融合：将视觉特征与文本引导对齐
        # 调整形状以适应注意力层 [B*T*H, W, C]
        visual_flat_a = visual_preds.reshape(-1, A, C) # [4, 16, 128, 128, 3] -> [4*16*128, 128, 3]
        text_flat_a = text_guide.reshape(-1, A, C)     # [4, 16, 128, 128, 3] -> [4*16*128, 128, 3]

        visual_flat_r = visual_preds.permute(0,1,3,2,4).reshape(-1, R, C) # [4, 16, 128, 128, 3] -> [4*16*128, 128, 3]
        text_flat_r = text_guide.permute(0,1,3,2,4).reshape(-1, R, C)     # [4, 16, 128, 128, 3] -> [4*16*128, 128, 3]

        # 计算注意力（文本引导视觉特征）
        fused_flat_a, _ = self.attention(visual_flat_a, text_flat_a, text_flat_a) # [4*16*128, 128, 3] - > [4*16*128, 128, 3]
        fused_flat_r, _ = self.attention(visual_flat_r, text_flat_r, text_flat_r) # [4*16*128, 128, 3] - > [4*16*128, 128, 3]
        
        # 恢复原始形状
        fused_preds_a = fused_flat_a.view(B, T, R, A, C)
        fused_preds_r = fused_flat_r.view(B, T, A, R, C).permute(0,1,3,2,4)
        
        # 残差连接，保留原始视觉信息
        return fused_preds_r, fused_preds_a, visual_preds, text_guide


# 方式2：文本-视觉对比损失（跨模态对齐）
class TextVisualContrastiveLoss(nn.Module):
    def __init__(self, class_dim=3, text_dim=512, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        # self.proj = nn.Linear(class_dim, text_dim)
        self.proj = nn.Linear(text_dim, class_dim)

    def forward(self, visual_preds, text_features):
        """
        计算视觉特征与文本特征的对比损失，使同类目标与文本描述对齐
        """
        B, T, H, W, C = visual_preds.shape
        text_features = text_features.to(torch.float32)
        # 提取视觉特征：对每个类别计算空间平均作为该类别的视觉特征
        visual_features = torch.mean(visual_preds, dim=(2, 3))  # [B, T, C]
        # 取每帧的平均作为视频级特征 [B, C]
        visual_features = visual_features.view(-1, C) # [B, T, C] -> [B*T, C] 
        # L2归一化
        visual_features = F.normalize(visual_features, dim=1)  # [B*T, C]
        
        # 文本特征已归一化，形状[B, 512]
        # 投影到相同维度（如果视觉特征维度≠512）
        if visual_features.shape[1] != text_features.shape[1]:
            # visual_features = self.proj(visual_features) # [B*T, 3] -> [B*T, 512]
            # visual_features = F.normalize(visual_features, dim=1)  # [B*T, 512]

            text_features = self.proj(text_features) # [B*T, 512] -> [B*T, 3]
            text_features = F.normalize(text_features, dim=1)  # [B*T, 3]
        
        # 计算文本-视觉相似度矩阵
        logits = torch.matmul(visual_features, text_features.t()) / self.temperature  # [B*T, B]
        
        # 对角线为正样本（同一batch的文本-视觉对）
        # labels = torch.arange(B, device=visual_features.device)
        labels = torch.eye(B*T, device=visual_features.device)

        # 双向交叉熵损失（类似CLIP）
        assert labels.shape == logits.shape
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss


# 方式3：文本引导的类别权重调整
class TextGuidedClassWeight(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=64, text_dim=512):
        super().__init__()

        # 根据文本特征动态生成类别权重
        self.weight_generator = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)  # 权重归一化
        )
        
    def forward(self, visual_preds, text_features):
        """
        参数:
            visual_preds: 每个类别的预测概率，形状[B, T, H, W, C]
            text_features: CLIP文本特征，形状[B, 512]
        返回:
            weighted_logits: 加权后的预测结果
        """
        B, T, H, W, C = visual_preds.shape
        text_features = text_features.to(torch.float32)
        # 生成类别权重 [B, C]
        class_weights_init = self.weight_generator(text_features)  # [B*T, 3]
        # [B*T, 3] -> [B, T, 3]
        class_weights_init = class_weights_init.view(B, T, -1)
        # 广播到所有帧和空间位置 [B, T, 1, 1, 3]
        class_weights = class_weights_init.unsqueeze(2).unsqueeze(3)
        
        # 应用权重：文本提到的类别权重更高
        weighted_preds = visual_preds * class_weights
        return weighted_preds, class_weights_init

def heatmap_to_batch_class_prob(heatmap_label, mode="mean"):
    """
    转换为(B, C)概率型输出（0~1之间，保留置信度）
    Args:
        heatmap_label: (B, T, H, W, C) 热力图标签
        mode: "mean"（时空均值）或 "max"（时空最大值）
    Returns:
        batch_prob: (B, C) 概率值
    """
    if mode == "mean":
        batch_prob = heatmap_label.mean(dim=(2, 3))  # 时空均值置信度
    elif mode == "max":
        batch_prob = heatmap_label.max(dim=(2, 3))[0]  # 时空最大置信度
    else:
        raise ValueError("mode must be 'mean' or 'max'")
    
    return batch_prob  # (B, C)，值越大表示样本包含该类的置信度越高

def heteroscedastic_loss(mean, log_var, target):
    """
    异方差损失函数，适用于不确定度估计
    
    参数:
    - mean: 预测均值 (B, T, H, W, C)
    - log_var: 预测对数方差 (B, T, H, W, C)
    - target: 目标值 (B, T, H, W, C)
    
    返回:
    - loss: 异方差损失值
    """
    # 计算方差
    var = torch.exp(log_var) # [4, 16, 128, 128, 3]
    
    # 计算加权MSE损失
    mse_loss = (target - mean) **2 / var # [4, 16, 128, 128, 3]
    var_loss = log_var
    
    # 总损失 (平均所有空间和时间维度), 加上一个小常数防止数值问题
    loss = 0.5 * (mse_loss + var_loss).mean() + 1e-6
    
    return loss 
    
def pad_tensor(input_tensor, target_shape=(4, 16, 224, 224, 3)):
    """
    将输入张量从(4,16,4,128,128,2)填充到(4,16,4,224,224,2)
    使用0进行填充
    """
    # 计算需要在高度和宽度维度上填充的数量
    pad_h = target_shape[2] - input_tensor.shape[2]  # 224 - 128 = 96
    pad_w = target_shape[3] - input_tensor.shape[3]  # 224 - 128 = 96
    
    # 在高度和宽度维度上均匀分配填充
    pad_h_left = pad_h // 2
    pad_h_right = pad_h - pad_h_left
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    
    # 应用填充 (只对最后两个空间维度进行填充)
    padded_tensor = torch.nn.functional.pad(
        input_tensor.permute(0,1,4,2,3), 
        (pad_w_left, pad_w_right, pad_h_left, pad_h_right),
        mode='constant', 
        value=0
    )
    
    return padded_tensor

def radar_and_text_encoding_function(outputs, text_describtion, config, clip_model, phase):

    chirps_length = len(config['dataset']['chirps'])
    vector_length = config['model']['clip_model']['clip_vector']
    is_random_chirp = config['dataset']['is_random_chirp']
    calc_all_window_size = config['dataset']['calc_all_window_size']

    text_encode = torch.zeros((1 if phase == 'valid' else config['train']['batch_size'],
                               1 if calc_all_window_size == False else config['model']['window_size'], 
                               1 if is_random_chirp == True else chirps_length,
                               vector_length
                             )).to(text_describtion.device)

    radar_encode = torch.zeros((1 if phase == 'valid' else config['train']['batch_size'],
                                1 if calc_all_window_size == False else config['model']['window_size'], 
                                1 if is_random_chirp == True else chirps_length,
                                vector_length
                                )).to(text_describtion.device)
    
    results_pad_predict = pad_tensor(outputs['output']) # [4, 16, 128, 128, 3] -> [4, 16, 3, 224, 224]
            
    if calc_all_window_size:
        for window_idx in range(config['model']['window_size']):
            if is_random_chirp:
                chirp_idx = random.randint(0, len(config['dataset']['chirps']) - 1)
                text_encode[:, window_idx, 0, :] = clip_model.encode_text(text_describtion[ :, window_idx, chirp_idx, :])
                radar_encode[:, window_idx, 0, :] = clip_model.encode_image(results_pad_predict[ :, window_idx, :, :, :]) 
            else:
                for chirp_idx in range(chirps_length):
                    text_encode[:, window_idx, chirp_idx, :] = clip_model.encode_text(text_describtion[ :, window_idx, chirp_idx, :])
                    radar_encode[ :, window_idx, chirp_idx, :] = clip_model.encode_image(results_pad_predict[ :, window_idx, :, :, :])
        if is_random_chirp:
            text_encode = text_encode.repeat(1,1,4,1)
    else:
        window_idx = random.randint(0, config['model']['window_size'] - 1)
        if is_random_chirp:
            chirp_idx = random.randint(0, len(config['dataset']['chirps']) - 1)
            text_encode[:, 0, 0, :] = clip_model.encode_text(text_describtion[ :, window_idx, chirp_idx, :])
            radar_encode[ :, 0, 0, :] = clip_model.encode_image(results_pad_predict[ :, window_idx, :, :, :])
            text_encode = text_encode.squeeze(1,2)
            radar_encode = radar_encode.squeeze(1,2)
        else:
            for chirp_idx in range(chirps_length):
                text_encode[:, 0, chirp_idx, :] = clip_model.encode_text(text_describtion[ :, window_idx, chirp_idx, :])
                radar_encode[ :, 0, chirp_idx, :, :, :] = clip_model.encode_image(results_pad_predict[ :, window_idx, :, :, :])

    return radar_encode, text_encode


def clip_contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    CLIP模型的对比损失函数实现
    
    参数:
        image_embeddings: 图像特征嵌入，形状为 [batch_size, embedding_dim]
        text_embeddings: 文本特征嵌入，形状为 [batch_size, embedding_dim]
        temperature: 温度参数，控制分布的尖锐程度，通常设置为0.07
    
    返回:
        loss: 对比损失值
    """
    # 确保输入是归一化的
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings  = F.normalize(text_embeddings, dim=1)
    
    # 计算图像和文本之间的相似度矩阵 (logits)
    # 形状为 [batch_size, batch_size]
    # logits_per_image = torch.matmul(image_embeddings, text_embeddings.t()) / temperature
    # logits_per_text = logits_per_image.t()  # 转置得到文本到图像的logits
    logits_per_image = (image_embeddings @ text_embeddings.T) / temperature
    logits_per_text = logits_per_image.T
    
    
    # 构建标签：对角线元素为正样本
    batch_size = image_embeddings.shape[0]
    # labels = torch.arange(batch_size, device=image_embeddings.device)
    labels = torch.eye(batch_size, device=image_embeddings.device)
    
    # 计算图像侧和文本侧的交叉熵损失
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    # 总损失为两侧损失的平均值
    total_loss = (loss_i + loss_t) / 2
    
    return total_loss

class PerClassCountingLoss(nn.Module):
    def __init__(self, epsilon=1e-6, weight=None):
        """
        每个类别的计数损失
        
        参数:
            epsilon: 防止除零的小值
            weight: 类别权重，形状为[3]，用于平衡不同类别的损失
        """
        super(PerClassCountingLoss, self).__init__()
        self.epsilon = epsilon
        # 初始化类别权重，默认为全1
        self.weight = torch.tensor(weight if weight is not None else [1.0, 1.0, 1.0])
        
    def forward(self, predictions, gt):
        """
        计算每个类别的计数损失
        
        参数:
            predictions: 网络输出，形状为[4, 16, 128, 128, 3] 
                         (batch, frames, height, width, classes)
            gt: 真实标签，形状同上
            
        返回:
            total_loss: 总损失
            class_losses: 每个类别的损失 [行人损失, 骑行者损失, 汽车损失]
        """
        # 如果输入是numpy数组，转换为torch张量
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions, dtype=torch.float32)
        if isinstance(gt, np.ndarray):
            gt = torch.tensor(gt, dtype=torch.float32)
            
        # 确保权重设备与输入一致
        self.weight = self.weight.to(predictions.device)
        
        # 计算每个类别的目标数量 (在空间维度求和)
        # 结果形状: [4, 16, 3]
        pred_counts = torch.sum(predictions, dim=(2, 3))
        gt_counts = torch.sum(gt, dim=(2, 3))
        
        # 计算L1损失 (绝对误差)
        l1_loss = torch.abs(pred_counts - gt_counts)
        
        # 计算相对误差损失 (相对比例误差)
        rel_loss = l1_loss / (gt_counts + self.epsilon)
        
        # 组合损失 (L1损失确保数值误差，相对损失确保比例误差)
        combined_loss = 0.5 * l1_loss + 0.5 * rel_loss
        
        # 按类别计算平均损失
        class_losses = torch.mean(combined_loss, dim=(0, 1))  # 在batch和frame维度求平均
        
        # 应用类别权重并计算总损失
        total_loss = torch.sum(class_losses * self.weight)
        
        return total_loss, class_losses