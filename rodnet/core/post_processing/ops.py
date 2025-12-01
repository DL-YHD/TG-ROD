import numpy as np


def detect_peaks(image, threshold=0.3):
    peaks_row = []
    peaks_col = []
    height, width = image.shape
    for h in range(1, height - 1):
        for w in range(2, width - 2):
            area = image[h - 1:h + 2, w - 2:w + 3]
            center = image[h, w]
            flag = np.where(area >= center)
            if flag[0].shape[0] == 1 and center > threshold:
                peaks_row.append(h)
                peaks_col.append(w)

    return peaks_row, peaks_col


def detect_multi_class_peaks(images, threshold=0.3):
    """
    批量检测图像中的峰值
    
    参数:
        images: 输入批量图像，形状为 (batch_size, num_frames, channels, height, width)
                即 (4, 16, 3, 128, 128)
        threshold: 峰值阈值
        
    返回:
        三维列表，结构为 [batch][frame][channel] = (peaks_row, peaks_col)
        其中每个元素是包含两个列表的元组，分别表示峰值的行坐标和列坐标
    """
    # 获取输入维度
    batch_size, num_frames, channels, height, width = images.shape
    
    # 初始化结果存储结构
    all_peaks = []
    
    # 遍历每个batch
    for b in range(batch_size):
        batch_peaks = []
        
        # 遍历每个帧
        for f in range(num_frames):
            frame_peaks = []
            
            # 遍历每个通道（类别）
            for c in range(channels):
                # 获取当前通道的图像
                img = images[b, f, c, :, :]
                peaks_row = []
                peaks_col = []
                
                # 检测峰值（保持原有的峰值检测逻辑）
                for h in range(1, height - 1):
                    for w in range(2, width - 2):
                        # 提取局部区域
                        area = img[h - 1:h + 2, w - 2:w + 3]
                        center = img[h, w]
                        
                        # 判断是否为峰值
                        flag = np.where(area >= center)
                        if flag[0].shape[0] == 1 and center > threshold:
                            peaks_row.append(h)
                            peaks_col.append(w)
                
                frame_peaks.append((peaks_row, peaks_col))
            
            batch_peaks.append(frame_peaks)
        
        all_peaks.append(batch_peaks)
    
    return all_peaks