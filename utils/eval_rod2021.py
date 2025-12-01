import os
import numpy as np

from .load_txt import read_gt_txt, read_sub_txt
from .rod_eval_utils import compute_ols_dts_gts, evaluate_img, accumulate, summarize

import matplotlib.pyplot as plt

olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)


def evaluate_rod2021(submit_dir, truth_dir, dataset):
    sub_names = sorted(os.listdir(submit_dir))
    gt_names = sorted(os.listdir(truth_dir))
    assert len(sub_names) == len(gt_names), "missing submission files!"
    for sub_name, gt_name in zip(sub_names, gt_names):
        if sub_name != gt_name:
            raise AssertionError("wrong submission file names!")

    # evaluation start
    evalImgs_all = []
    n_frames_all = 0
    ols_list = []

    for seqid, (sub_name, gt_name) in enumerate(zip(sub_names, gt_names)):
        gt_path = os.path.join(truth_dir, gt_name)
        sub_path = os.path.join(submit_dir, sub_name)
        data_path = os.path.join(dataset.data_root, 'sequences', 'valid', gt_names[seqid][:-4])
        n_frame = len(os.listdir(os.path.join(data_path, dataset.sensor_cfg.camera_cfg['image_folder'])))

        gt_dets = read_gt_txt(gt_path, n_frame, dataset)
        sub_dets = read_sub_txt(sub_path, n_frame, dataset)

        olss_all = {(imgId, catId): compute_ols_dts_gts(gt_dets, sub_dets, imgId, catId, dataset) \
                    for imgId in range(n_frame)
                    for catId in range(3)}

        for olss in list(olss_all.values()):
            if len(olss) > 0:
                olss_max_gt = np.amax(olss, axis=0)
                cur_olss = list(np.ravel(np.squeeze(olss_max_gt)))
                ols_list.extend(cur_olss)

        evalImgs = [evaluate_img(gt_dets, sub_dets, imgId, catId, olss_all, olsThrs, recThrs, dataset)
                    for imgId in range(n_frame)
                    for catId in range(3)]

        n_frames_all += n_frame
        evalImgs_all.extend(evalImgs)

    class_AP = np.zeros(4)
    class_AR = np.zeros(4)
    

    class_thrs_AP = np.zeros([4, 6])
    class_thrs_AR = np.zeros([4, 6])
    
    # # 类别名称（第一维：4个类别）
    # class_names = ["total", "pedestrian", "cyclist", "car"]
    # # OLS阈值（第二维：6个阈值）
    # ols_thresholds = [0, 0.5, 0.6, 0.7, 0.8, 0.9]

    # colors = ['blue', 'orange', 'green', 'purple']
    # linestyles = ['--', '-.', '-', ':']
    # plt.figure(figsize=(8, 6))

    for class_id in range(4):
        eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, dataset, log=False, class_id = class_id)

        stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
        stats_thrs = summarize(eval, olsThrs, recThrs, dataset, gl=True)

        class_AP[class_id] = stats[0]
        class_AR[class_id] = stats[1]

        class_thrs_AP[class_id] = np.array([stats_thrs[0], stats_thrs[1], stats_thrs[2], stats_thrs[3],  stats_thrs[4], stats_thrs[5]])
        class_thrs_AR[class_id] = np.array([stats_thrs[6], stats_thrs[7], stats_thrs[8], stats_thrs[9], stats_thrs[10], stats_thrs[11]])

        # plt.plot(ols_thresholds[1:], class_thrs_AP[class_id][1:] * 100, color=colors[class_id], linestyle=linestyles[class_id], linewidth=2, label=f'{class_names[class_id]}')
    
    # # 图表美化
    # plt.xlabel('OLS Threshold', fontsize=12)
    # plt.ylabel('AP on CRUW Validation Set (%)', fontsize=12)
    # # plt.title('AP vs. OLS Threshold', fontsize=14, fontweight='bold')
    # plt.legend(loc='upper right', fontsize=10)
    # plt.grid(True, alpha=0.3)
    # plt.xlim([0.8, 0.9])
    # plt.xticks(np.arange(0.8, 1.0, 0.05))
    # plt.ylim([50, 100])
    # plt.yticks(np.arange(50, 110, 10))
    # # 填充区域（可选，模拟原图的背景区分）
    # plt.axvspan(0.5, 0.9, color='red', alpha=0.1)
    # plt.axvspan(0, 0.5, color='blue', alpha=0.1)
    # plt.tight_layout()
    # plt.show()
    # # plt.show()
    # plt.savefig('/media/user/WDC10TB/code/YHD/code/mRadNet-main/submits/AP_OLS.jpg')
    # plt.clf()
    print('class_thrs_AP = ', class_thrs_AP)
    print('class_thrs_AR = ', class_thrs_AR)


    # print("AP_total: %.4f | AP_pedestrian: %.4f | AP_cyclist: %.4f | AP_car: %.4f" % (class_AP[3], class_AP[0], class_AP[1], class_AP[2]))
    # print("AR_total: %.4f | AR_pedestrian: %.4f | AR_cyclist: %.4f | AR_car: %.4f" % (class_AR[3], class_AR[0], class_AR[1], class_AR[2]))

    return class_AP, class_AR
