import argparse
import os,glob
from datasets.siim import get_loader
from torch.backends import cudnn
import random
import json,codecs
from pprint import pprint
from utils.mask_functions import write_txt
from utils.datasets_statics import DatasetsStatic
from argparse import Namespace
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pickle
from datetime import datetime

FREEZE = False
if FREEZE:
    from solver_freeze import Train
else:
    from solver import Train


def main(config):
    cudnn.benchmark = True

    config.save_path = config.model_path + '/' + config.model_type
    if not os.path.exists(config.save_path):
        print('Making pth folder...')
        os.makedirs(config.save_path)

    # 打印配置参数，并输出到文件中
    pprint(config)
    if 'choose_threshold' not in config.mode:
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now()) 
        with codecs.open(config.save_path + '/'+ TIMESTAMP + '.json', 'w', "utf-8") as json_file:
            json.dump({k: v for k, v in config._get_kwargs()}, json_file, ensure_ascii=False)
    # write_txt(config.save_path, {k: v for k, v in config._get_kwargs()})

    # 存储每一次交叉验证的最高得分，最优阈值
    scores, best_thrs, best_pixel_thrs = [], [], []

    # 统计各样本是否有Mask
    if os.path.exists('dataset_static.pkl'):
        print('Extract dataset static information form: dataset_static.pkl.')
        with open('dataset_static.pkl', 'rb') as f:
            static = pickle.load(f)
            images_path, masks_path, masks_bool = static[0], static[1], static[2]
    else:
        print('Calculate dataset static information.')
        # 为了确保每次重新运行，交叉验证每折选取的下标均相同(因为要选阈值),以及交叉验证的种子固定。
        dataset_static = DatasetsStatic(config.dataset_root, 'train_images', 'train_mask', True)
        images_path, masks_path, masks_bool = dataset_static.mask_static_bool()
        with open('dataset_static.pkl', 'wb') as f:
            pickle.dump([images_path, masks_path, masks_bool], f)
    
    # 统计各样本是否有Mask
    if os.path.exists('dataset_static_mask.pkl'):
        print('Extract dataset static information form: dataset_static_mask.pkl.')
        with open('dataset_static_mask.pkl', 'rb') as f:
            static_mask = pickle.load(f)
            images_path_mask, masks_path_mask, masks_bool_mask = static_mask[0], static_mask[1], static_mask[2]
    else:
        print('Calculate dataset with mask static information.')
        # 为了确保每次重新运行，交叉验证每折选取的下标均相同(因为要选阈值),以及交叉验证的种子固定。
        dataset_static_mask = DatasetsStatic(config.dataset_root, 'train_images', 'train_mask', True)
        images_path_mask, masks_path_mask, masks_bool_mask = dataset_static_mask.mask_static_bool_stage3()
        with open('dataset_static_mask.pkl', 'wb') as f:
            pickle.dump([images_path_mask, masks_path_mask, masks_bool_mask], f)

    result = {}
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=1)
    split1, split2 = skf.split(images_path, masks_bool), skf.split(images_path_mask, masks_bool_mask)
    for index, ((train_index, val_index), (train_index_mask, val_index_mask)) in enumerate(zip(split1, split2)):
        # if index > 1:    if index < 2 or index > 3:    if index < 4:
        # 不管是选阈值还是训练，均需要对下面几句话进行调整，来选取测试哪些fold。另外，选阈值的时候，也要对choose_threshold参数更改(是否使用best)
        # if index != 0:
        #     print("Fold {} passed".format(index))
        #     continue
        train_image = [images_path[x] for x in train_index]
        train_mask = [masks_path[x] for x in train_index]
        val_image = [images_path[x] for x in val_index]
        val_mask = [masks_path[x] for x in val_index]

        train_image_mask = [images_path_mask[x] for x in train_index_mask]
        train_mask_mask = [masks_path_mask[x] for x in train_index_mask]
        val_image_mask = [images_path_mask[x] for x in val_index_mask]
        val_mask_mask = [masks_path_mask[x] for x in val_index_mask]

        # 对于第一个阶段方法的处理
        train_loader, val_loader = get_loader(train_image, train_mask, val_image, val_mask, config.image_size_stage1,
                                        config.batch_size_stage1, config.num_workers, config.stage1_augmentation_flag, weights_sample=config.weight_sample)
        solver = Train(config, train_loader, val_loader)
        # 针对不同mode，在第一阶段的处理方式
        if config.mode == 'train' or config.mode == 'train_stage1':
            solver.train(index)
        elif config.mode == 'choose_threshold1':
            best_thr, best_pixel_thr, score = solver.choose_threshold(os.path.join(config.save_path, '%s_%d_%d_best.pth' % (config.model_type, 1, index)), index)
            scores.append(score)
            best_thrs.append(best_thr)
            best_pixel_thrs.append(best_pixel_thr)
            result[str(index)] = [best_thr, best_pixel_thr, score]
        del train_loader, val_loader

        # 对于第二个阶段的处理方法
        train_loader_stage2, val_loader_stage2 = get_loader(train_image, train_mask, val_image, val_mask, config.image_size_stage2,
                                    config.batch_size_stage2, config.num_workers, config.stage2_augmentation_flag, weights_sample=config.weight_sample)
        # 更新类的训练集以及验证集
        solver.train_loader, solver.valid_loader = train_loader_stage2, val_loader_stage2
        # 针对不同mode，在第二阶段的处理方式
        if config.mode == 'train' or config.mode == 'train_stage2' or config.mode == 'train_stage23':
            solver.train_stage2(index)
        elif config.mode == 'choose_threshold2':
            # solver.pred_mask_count(os.path.join(config.save_path, '%s_%d_%d_best.pth' % (config.model_type, 2, index)), masks_bool, val_index, 0.80, 1280)
            best_thr, best_pixel_thr, score = solver.choose_threshold_grid(os.path.join(config.save_path, '%s_%d_%d_best.pth' % (config.model_type, 2, index)), index)
            scores.append(score)
            best_thrs.append(best_thr)
            best_pixel_thrs.append(best_pixel_thr)
            result[str(index)] = [best_thr, best_pixel_thr, score]
        del train_loader_stage2, val_loader_stage2

        # 对于第三个阶段的处理方法
        # 第三阶段和第二阶段使用的图片大小一致，最大batch_size一致
        train_loader_stage3, val_loader_stage3 = get_loader(train_image_mask, train_mask_mask, val_image_mask, val_mask_mask, config.image_size_stage2,
                                    config.batch_size_stage2, config.num_workers, config.stage3_augmentation_flag, weights_sample=config.weight_sample)
        # 更新类的训练集以及验证集
        solver.train_loader, solver.valid_loader = train_loader_stage3, val_loader_stage3        
        # 针对不同mode，在第三阶段的处理方式
        if config.mode == 'train' or config.mode == 'train_stage3' or config.mode == 'train_stage23':
            solver.train_stage3(index)
        elif config.mode == 'choose_threshold3':
            # solver.pred_mask_count(os.path.join(config.save_path, '%s_%d_%d_best.pth' % (config.model_type, 3, index)), masks_bool_mask, val_index_mask, 0.67, 0)
            best_thr, best_pixel_thr, score = solver.choose_threshold(os.path.join(config.save_path, '%s_%d_%d_best.pth' % (config.model_type, 3, index)), index)
            scores.append(score)
            best_thrs.append(best_thr)
            best_pixel_thrs.append(best_pixel_thr)
            result[str(index)] = [best_thr, best_pixel_thr, score]

    # 若为选阈值操作，则输出n_fold折验证集结果的平均值
    if 'choose_threshold' in config.mode:
        score_mean = np.array(scores).mean()
        thr_mean = np.array(best_thrs).mean()
        pixel_thr_mean = np.array(best_pixel_thrs).mean()
        print('score_mean:{}, thr_mean:{}, pixel_thr_mean:{}'.format(score_mean, thr_mean, pixel_thr_mean))
        result['mean'] = [float(thr_mean), float(pixel_thr_mean), float(score_mean)]

        with codecs.open(config.save_path + '/result_stage{}.json'.format(config.mode[-1]), 'w', "utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False)
        print('save the result')


if __name__ == '__main__':
    use_paras = False
    if use_paras:
        with open('./checkpoint/unet_resnet34/' + "params.json", 'r', encoding='utf-8') as json_file:
            config = json.load(json_file)
        # dict to namespace
        config = Namespace(**config)
    else:
        parser = argparse.ArgumentParser()
        '''
        第一阶段为768，第二阶段为1024，unet_resnet34时各个电脑可以设置的最大batch size
        zdaiot:10,6 z840:12,6 mxq:20,10
        '''
        parser.add_argument('--image_size_stage1', type=int, default=768, help='image size in the first stage')
        parser.add_argument('--batch_size_stage1', type=int, default=20, help='batch size in the first stage')
        parser.add_argument('--epoch_stage1', type=int, default=40, help='How many epoch in the first stage')
        parser.add_argument('--epoch_stage1_freeze', type=int, default=0, help='How many epoch freezes the encoder layer in the first stage')

        parser.add_argument('--image_size_stage2', type=int, default=1024, help='image size in the second stage')
        parser.add_argument('--batch_size_stage2', type=int, default=10, help='batch size in the second stage')
        parser.add_argument('--epoch_stage2', type=int, default=15, help='How many epoch in the second stage')
        parser.add_argument('--epoch_stage2_accumulation', type=int, default=0, help='How many epoch gradients accumulate in the second stage')
        parser.add_argument('--accumulation_steps', type=int, default=10, help='How many steps do you add up to the gradient in the second stage')

        parser.add_argument('--epoch_stage3', type=int, default=10, help='How many epoch in the third stage')
        parser.add_argument('--epoch_stage3_accumulation', type=int, default=0, help='How many epoch gradients accumulate in the third stage')

        parser.add_argument('--stage1_augmentation_flag', type=bool, default=True, help='if true, use augmentation method in stage1 train set')
        parser.add_argument('--stage2_augmentation_flag', type=bool, default=True, help='if true, use augmentation method in stage2 train set')
        parser.add_argument('--stage3_augmentation_flag', type=bool, default=True, help='if true, use augmentation method in stage3 train set')
        parser.add_argument('--n_splits', type=int, default=5, help='n_splits_fold')

        # model set 
        parser.add_argument('--resume', type=str, default='', help='if has value, must be the name of Weight file.')
        '''mode可选值 没有考虑各自阶段训练到一半重新加载的情况，因为学习率为余弦衰减，不可控 TODO
        train: 训练所有阶段, resume必须为空
        train_stage1: 只训练第一阶段, resume必须为空
        train_stage2: 只训练第二阶段，resume不能为空
        train_stage3: 只训练第三阶段，resume不能为空
        train_stage23: 只训练第二和第三阶段，resume不能为空
        choose_threshold1: 只选第一阶段的阈值
        choose_threshold2: 只选第二阶段的阈值
        choose_threshold3: 只选第三阶段的阈值
        '''
        parser.add_argument('--mode', type=str, default='train', \
            help='train/train_stage1/train_stage2/train_stage3/train_stage23/choose_threshold1/choose_threshold2/choose_threshold3.')
        parser.add_argument('--model_type', type=str, default='unet_resnet34', \
            help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/unet_resnet34/linknet/deeplabv3plus/pspnet_resnet34/unet_se_resnext50_32x4d/unet_densenet121')

        # model hyper-parameters
        parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
        parser.add_argument('--img_ch', type=int, default=3)
        parser.add_argument('--output_ch', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--lr', type=float, default=2e-4, help='init lr in stage1')
        parser.add_argument('--lr_stage2', type=float, default=5e-6, help='init lr in stage2')
        parser.add_argument('--lr_stage3', type=float, default=1e-7, help='init lr in stage3')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay in optimizer')
        
        # dataset 
        parser.add_argument('--model_path', type=str, default='./checkpoints')
        parser.add_argument('--dataset_root', type=str, default='./datasets/SIIM_data')
        parser.add_argument('--train_path', type=str, default='./datasets/SIIM_data/train_images')
        parser.add_argument('--mask_path', type=str, default='./datasets/SIIM_data/train_mask')
        parser.add_argument('--weight_sample', type=list, default=0, help='sample weight of class')

        config = parser.parse_args()
        # config = {k: v for k, v in args._get_kwargs()}

    if config.mode == 'train_stage2' or config.mode == 'train_stage3' or config.mode == 'train_stage23':
        assert config.resume != ''
    elif config.mode == 'train' or config.mode == 'train_stage1':
        assert config.resume == ''
    main(config)
