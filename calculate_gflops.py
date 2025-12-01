import torch
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile

from model.mRadNet import mRadNet
import yaml

def calculate_fps(model, inputs, device='cuda', num_warmup=10, num_test=100):
    """
    计算模型推理FPS
    参数：
        model : 待测试模型
        input_size : 输入张量尺寸 (batch, channel, height, width)
        device : 测试设备 cuda/cpu
        num_warmup : 预热次数
        num_test : 正式测试次数
    """
    model.to(device)
    model.eval()


    # 预热阶段
    with torch.no_grad():
        for _ in range(num_warmup):
            # _ = model(inputs[0],inputs[1])
            _ = model(inputs[0])

    # CUDA同步计时
    torch.cuda.synchronize()
    start_time = time.time()

    # 正式测试
    with torch.no_grad():
        for _ in range(num_test):
            # _ = model(inputs[0],inputs[1])
            _ = model(inputs[0])

    # CUDA同步计时
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # 计算指标
    avg_time = elapsed / num_test
    fps = 1.0 / avg_time

    print(f"Average inference time: {avg_time * 1000:.2f}ms")
    print(f"FPS: {fps:.2f}")
    return fps



if __name__ == '__main__':
    """model inreduction
        最前端Q+PW卷积
    """

    test_flops = True
    with open('config/mRadNet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # [B, T, R, A, C=2*chirps]

    # image = torch.rand((1, 3, 16, 300, 500), device='cpu')
    radar = torch.rand((1, 16, 4, 128, 128, 2), device='cuda')
    # image = torch.rand((1, 3, 5, 300, 256), device='cuda')
    # ra    = torch.rand((1, 1, 5, 256, 256), device='cuda')
    
    device = torch.device('cuda:0')

    net = mRadNet(
        model_cfg=config['model'],
        dataset_cfg=config['dataset']
    ).to(device)

    if test_flops:
    # FLOPs = 27.6G
    # params = 148.3M
        flops, params = profile(net, inputs=(radar, ))
        print("FLOPs=", str((flops) / 1e9) + '{}'.format("G"))
        print("params=", str(params / 1e6) + '{}'.format("M"))
    else:
        calculate_fps(net,
                    inputs=(radar,),  # 根据模型输入调整
                    device=device,
                    num_test=100)

    
    # calculate_fps(net,
    #             inputs=(radar,),  # 根据模型输入调整
    #             device=device,
    #             num_test=100)
    # # FLOPs = 27.6G
    # # params = 148.3M
    # flops, params = profile(net, inputs=(radar, ))
    # print("FLOPs=", str((flops) / 1e9) + '{}'.format("G"))
    # print("params=", str(params / 1e6) + '{}'.format("M"))


    # print(parameter_count_table(net))
