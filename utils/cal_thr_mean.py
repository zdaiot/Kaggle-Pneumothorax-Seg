import json, codecs
import numpy as np


def cal_thr_mean(file_name, model_name='unet_resnet34'):
    '''
    为了加快测试速度，有的时候使用多台电脑选阈值，这个时候需要手动计算阈值均值和像素阈值均值
    '''
    with open('checkpoints/'+model_name+file_name, 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    n_splits = [0, 2, 3, 4] # 
    thresholds, less_than_sum, score = [], [], []
    for x in n_splits:
        thresholds.append(config[str(x)][0])
        less_than_sum.append(config[str(x)][1])
        score.append(config[str(x)][2])

    thresholds_mean, less_than_sum_mean, score_mean = np.array(thresholds).mean(), np.array(less_than_sum).mean(), np.array(score).mean()
    config['mean'] = [float(thresholds_mean), float(less_than_sum_mean), float(score_mean)]
    print(config)

    with codecs.open('checkpoints/'+model_name+file_name, 'w', "utf-8") as json_file:
        json.dump(config, json_file, ensure_ascii=False)

if __name__ == "__main__":
    file_name = '/result_stage2.json'
    cal_thr_mean(file_name)

    file_name = '/result_stage3.json'
    cal_thr_mean(file_name)

    file_name = '/result_stage2_score.json'
    cal_thr_mean(file_name)