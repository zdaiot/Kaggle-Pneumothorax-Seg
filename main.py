import argparse
import os,glob
from solver import Train
from datasets.siim import get_loader
from torch.backends import cudnn
import random
import json,codecs
from pprint import pprint
from utils.mask_functions import write_txt
from argparse import Namespace
from sklearn.model_selection import KFold
import numpy as np


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net', 'unet_resnet34']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/unet_resnet34')
        print('Your input for model_type was %s' % config.model_type)
        return

    # 配置随机学习率，随机权重衰减，保存路径等
    lr = random.random() * 0.0005 + 0.0000005
    augmentation_prob = random.random() * 0.7
    decay_ratio = random.random() * 0.8
    decay_epoch = int((config.epoch_stage1+config.epoch_stage2) * decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.lr = lr
    config.num_epochs_decay = decay_epoch
    config.save_path = config.model_path + '/' + config.model_type
    if not os.path.exists(config.save_path):
        print('Making pth folder...')
        os.makedirs(config.save_path)

    # 打印配置参数，并输出到文件中
    pprint(config)
    with codecs.open(config.save_path + '/params.json', 'w', "utf-8") as json_file:
        json.dump({k: v for k, v in config._get_kwargs()}, json_file, ensure_ascii=False)
    json_file.close()
    # write_txt(config.save_path, {k: v for k, v in config._get_kwargs()})

    # 开始，交叉验证
    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=1)
    # 存储每一次交叉验证的得分，最优阈值
    scores, best_thrs = [], []
    images_path = glob.glob(config.train_path+'/*.jpg')
    masks_path = glob.glob(config.mask_path+'/*.png')
    for index, (train_index, val_index) in enumerate(kf.split(images_path)):
        train_image = [images_path[x] for x in train_index]
        train_mask = [masks_path[x] for x in train_index]
        val_image = [images_path[x] for x in val_index]
        val_mask = [masks_path[x] for x in val_index]

        # 对于第一个阶段方法的处理
        train_loader, val_loader = get_loader(train_image, train_mask, val_image, val_mask, config.image_size_stage1,
                                        config.batch_size_stage1, config.num_workers, config.augmentation_flag)
        solver = Train(config, train_loader, valid_loader=val_loader, test_loader=None)
        # 是否选取阈值
        if config.mode == 'train':
            solver.train()
        else:
            # 是否为两阶段法，若为两阶段法，则pass，否则选取阈值
            if config.two_stage == False:
                score, best_thr = solver.choose_threshold(os.path.join(config.save_path, '%s_%d.pth' % (config.model_type, config.epoch_stage1)), index)
                scores.append(score)
                best_thrs.append(best_thr)
            else:
                pass

        # 对于第二个阶段的处理方法
        if config.two_stage == True:
            del train_loader, val_loader
            train_loader_, val_loader_ = get_loader(config.mask_path, config.train_path, config.val_path, config.image_size_stage2,
                                        config.batch_size_stage2, config.num_workers, config.augmentation_flag)
            # 更新类的训练集以及验证集
            solver.train_loader, solver.val_loader = train_loader_, val_loader_
            if config.mode == 'train':
                solver.train_stage2()
            else:
                score, best_thr = solver.choose_threshold(os.path.join(config.save_path, '%s_%d.pth' % (config.model_type, config.epoch_stage2+config.epoch_stage1)), index)
                scores.append(score)
                best_thrs.append(best_thr)
    
    # 若为选阈值操作，则输出n_fold折验证集结果的平均值
    if config.mode == 'choose_threshold':
        score_mean = np.array(scores).mean()
        thrs_mean = np.array(best_thrs).mean()
        print('score_mean:{}, thrs_mean:{}'.format(score_mean, thrs_mean))

        
if __name__ == '__main__':
    use_paras = False
    if use_paras:
        with open('./checkpoint/' + "params.json", 'r', encoding='utf-8') as json_file:
            config = json.load(json_file)
            json_file.close()
        # dict to namespace
        config = Namespace(**config)
    else:
        parser = argparse.ArgumentParser()
        # parser.add_argument('--image_size', type=int, default=512)
        # stage set，注意若当two_stage等于False的时候，epoch_stage2必须等于0，否则会影响到学习率衰减。其余参数以stage1的配置为准
        # 当save_step为10时，epoch_stage1和epoch_stage2必须是10的整数
        parser.add_argument('--two_stage', type=bool, default=False, help='if true, use two_stage method')
        parser.add_argument('--image_size_stage1', type=int, default=512, help='image size in the first stage')
        parser.add_argument('--batch_size_stage1', type=int, default=20, help='batch size in the first stage')
        parser.add_argument('--epoch_stage1', type=int, default=200, help='How many epoch in the first stage')
        parser.add_argument('--epoch_stage1_freeze', type=int, default=0, help='How many epoch freezes the encoder layer in the first stage')

        parser.add_argument('--image_size_stage2', type=int, default=1024, help='image size in the second stage')
        parser.add_argument('--batch_size_stage2', type=int, default=2, help='batch size in the second stage')
        parser.add_argument('--epoch_stage2', type=int, default=0, help='How many epoch in the second stage')
        parser.add_argument('--epoch_stage2_accumulation', type=int, default=15, help='How many epoch gradients accumulate in the second stage')
        parser.add_argument('--accumulation_steps', type=int, default=10, help='How many steps do you add up to the gradient in the second stage')

        parser.add_argument('--augmentation_flag', type=bool, default=True, help='if true, use augmentation method')
        parser.add_argument('--n_splits', type=int, default=5, help='n_splits_fold')

        # model set
        parser.add_argument('--resume', type=str, default='', help='if has value, must be the name of Weight file. Only work for first stage')
        parser.add_argument('--mode', type=str, default='train', help='train/choose_threshold')
        parser.add_argument('--model_type', type=str, default='unet_resnet34',
                            help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/unet_resnet34')

        # model hyper-parameters
        parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
        parser.add_argument('--img_ch', type=int, default=3)
        parser.add_argument('--output_ch', type=int, default=1)
        parser.add_argument('--num_epochs_decay', type=int, default=70) # TODO
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
        parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
        parser.add_argument('--augmentation_prob', type=float, default=0.4) # TODO
        parser.add_argument('--save_step', type=int, default=10)
        
        # dataset 
        parser.add_argument('--model_path', type=str, default='./checkpoints')
        parser.add_argument('--train_path', type=str, default='./datasets/SIIM_data/train_images')
        parser.add_argument('--mask_path', type=str, default='./datasets/SIIM_data/train_mask')
        parser.add_argument('--val_path', type=str, default='./datasets/SIIM_data')

        config = parser.parse_args()
        # config = {k: v for k, v in args._get_kwargs()}
    if config.two_stage == False:
        assert config.epoch_stage2 == 0,'当two_stage等于False的时候，epoch_stage2必须等于0，否则会影响到学习率衰减'
    main(config)
