import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ===================== 全局配置（优化图表显示）=====================
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 支持中文
rcParams['axes.unicode_minus'] = False  # 正确显示负号
rcParams['figure.dpi'] = 100  # 图表分辨率
rcParams['lines.linewidth'] = 2.5  # 线条宽度
rcParams['axes.labelsize'] = 12  # 坐标轴标签大小
rcParams['axes.titlesize'] = 14  # 标题大小
rcParams['legend.fontsize'] = 11  # 图例大小
rcParams['xtick.labelsize'] = 10  # x轴刻度大小
rcParams['ytick.labelsize'] = 10  # y轴刻度大小

# ===================== 数据定义（替换为你的实际数据）=====================
# 类别名称（第一维：4个类别）
class_names = ["整体", "行人", "骑行者", "汽车"]
# OLS阈值（第二维：6个阈值）
ols_thresholds = [0, 0.5, 0.6, 0.7, 0.8, 0.9]

# AP数据（形状：(4,6)，对应[类别, 阈值]）
class_thrs_AP = np.array([
    [75.2, 73.5, 72.1, 70.3, 65.8, 52.4],  # 整体AP
    [80.5, 78.3, 76.2, 74.1, 70.5, 55.2],  # 行人AP
    [70.1, 68.4, 66.3, 64.2, 60.1, 45.3],  # 骑行者AP
    [72.3, 70.2, 68.5, 66.4, 62.3, 50.1]   # 汽车AP
])

# AR数据（形状：(4,6)，对应[类别, 阈值]）
class_thrs_AR = np.array([
    [88.5, 86.3, 84.2, 81.5, 76.3, 65.2],  # 整体AR
    [90.2, 88.1, 85.3, 82.4, 78.2, 68.5],  # 行人AR
    [85.3, 83.2, 81.1, 78.3, 73.2, 62.1],  # 骑行者AR
    [86.4, 84.1, 82.3, 79.5, 74.3, 63.2]   # 汽车AR
])

# ===================== 图表绘制 =====================
def plot_ap_ar_curve(class_names, ols_thresholds, ap_data, ar_data):
    """
    绘制AP和AR随OLS阈值变化的曲线
    
    参数:
        class_names: 类别名称列表
        ols_thresholds: OLS阈值列表
        ap_data: AP数据数组 (4,6)
        ar_data: AR数据数组 (4,6)
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 定义颜色和线型（AP用实线，AR用虚线，不同类别不同颜色）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 专业配色
    ap_linestyles = ['-', '-', '-', '-']  # AP统一实线
    ar_linestyles = ['--', '--', '--', '--']  # AR统一虚线
    
    # 绘制AP曲线
    for i in range(len(class_names)):
        ax.plot(
            ols_thresholds,
            ap_data[i],
            color=colors[i],
            linestyle=ap_linestyles[i],
            marker='o',  # 标记点
            markersize=4,
            label=f'{class_names[i]} (AP)'
        )
    
    # 绘制AR曲线
    for i in range(len(class_names)):
        ax.plot(
            ols_thresholds,
            ar_data[i],
            color=colors[i],
            linestyle=ar_linestyles[i],
            marker='s',  # 标记点（与AP区分）
            markersize=4,
            label=f'{class_names[i]} (AR)'
        )
    
    # 图表美化
    ax.set_xlabel('OLS 阈值', fontsize=13)
    ax.set_ylabel('指标值 (%)', fontsize=13)
    ax.set_title('不同OLS阈值下各类型目标的AP和AR性能', fontsize=15, fontweight='bold')
    
    # 设置坐标轴范围
    ax.set_xlim([-0.05, 0.95])  # 左右留边
    ax.set_ylim([40, 100])       # 纵轴范围（根据数据调整）
    
    # 网格线（虚线，增加透明度）
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # 背景分区（不同阈值区间不同颜色，增强可读性）
    ax.axvspan(-0.05, 0.5, color='lightblue', alpha=0.1, label='低阈值区间')
    ax.axvspan(0.5, 0.95, color='lightcoral', alpha=0.1, label='高阈值区间')
    
    # 图例（分两列显示，避免拥挤）
    ax.legend(ncol=2, loc='lower left', bbox_to_anchor=(0, 0))
    
    # 调整布局（防止标签被截断）
    plt.tight_layout()
    
    # 保存图片（可选，支持高分辨率）
    plt.savefig('ap_ar_vs_ols_threshold.png', dpi=300, bbox_inches='tight')
    # plt.show()

# ===================== 数据验证与运行 =====================
if __name__ == "__main__":
    # 验证数据形状
    assert ap_data.shape == (4, 6), f"AP数据形状错误，应为(4,6)，实际为{ap_data.shape}"
    assert ar_data.shape == (4, 6), f"AR数据形状错误，应为(4,6)，实际为{ar_data.shape}"
    assert len(class_names) == 4, f"类别数错误，应为4，实际为{len(class_names)}"
    assert len(ols_thresholds) == 6, f"阈值数错误，应为6，实际为{len(ols_thresholds)}"
    
    # 绘制图表
    plot_ap_ar_curve(class_names, ols_thresholds, class_thrs_AP, class_thrs_AR)
    
    # 输出数据汇总表
    print("="*60)
    print("各阈值下AP/AR性能汇总")
    print("="*60)
    for thr_idx, thr in enumerate(ols_thresholds):
        print(f"\nOLS阈值 = {thr}:")
        print(f"{'类别':<8} {'AP(%)':<10} {'AR(%)':<10}")
        print("-"*30)
        for cls_idx, cls in enumerate(class_names):
            ap_val = class_thrs_AP[cls_idx, thr_idx]
            ar_val = class_thrs_AR[cls_idx, thr_idx]
            print(f"{cls:<8} {ap_val:<10.2f} {ar_val:<10.2f}")
    print("="*60)