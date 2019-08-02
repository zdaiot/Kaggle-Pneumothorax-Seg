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
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net', 'unet_resnet34', 'linknet', 'deeplabv3plus', 'pspnet_resnet34']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/unet_resnet34')
        print('Your input for model_type was %s' % config.model_type)
        return

    # 配置随机权重衰减，保存路径等
    decay_ratio = random.random() * 0.8
    decay_epoch = int((config.epoch_stage1+config.epoch_stage2) * decay_ratio)
    config.num_epochs_decay = decay_epoch

    config.save_path = config.model_path + '/' + config.model_type
    if not os.path.exists(config.save_path):
        print('Making pth folder...')
        os.makedirs(config.save_path)

    # 打印配置参数，并输出到文件中
    pprint(config)
    if config.mode != 'choose_threshold':
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
        images_path, masks_path, masks_bool = dataset_static.mask_static_level(level=5)
        with open('dataset_static.pkl', 'wb') as f:
            pickle.dump([images_path, masks_path, masks_bool], f)

    result = {}
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=1)
    for index, (train_index, val_index) in enumerate(skf.split(images_path, masks_bool)):
        # if index > 1:    if index < 2 or index > 3:    if index < 4:
        # 不管是选阈值还是训练，均需要对下面几句话进行调整，来选取测试哪些fold。另外，选阈值的时候，也要对choose_threshold参数更改(是否使用best)
        if index != 0:
            print("Fold {} passed".format(index))
            continue
        train_image = [images_path[x] for x in train_index]
        train_mask = [masks_path[x] for x in train_index]
        val_image = [images_path[x] for x in val_index]
        val_mask = [masks_path[x] for x in val_index]

        # 对于第一个阶段方法的处理
        train_loader, val_loader = get_loader(train_image, train_mask, val_image, val_mask, config.image_size_stage1,
                                        config.batch_size_stage1, config.num_workers, config.augmentation_flag)
        solver = Train(config, train_loader, val_loader)
        # 针对不同mode，在第一阶段的处理方式
        if config.mode == 'train':
            solver.train(index)
        elif config.mode == 'train_stage2':
            pass
        else:
            # 是否为两阶段法，若为两阶段法，则pass，否则选取阈值
            if config.two_stage == False:
                best_thr, best_pixel_thr, score = solver.choose_threshold(os.path.join(config.save_path, '%s_%d_%d_best.pth' % (config.model_type, 1, index)), index)
                scores.append(score)
                best_thrs.append(best_thr)
                best_pixel_thrs.append(best_pixel_thr)
                result[str(index)] = [best_thr, best_pixel_thr, score]
            else:
                pass

        # 对于第二个阶段的处理方法
        if config.two_stage == True:
            del train_loader, val_loader
            train_loader_, val_loader_ = get_loader(train_image, train_mask, val_image, val_mask, config.image_size_stage2,
                                        config.batch_size_stage2, config.num_workers, config.augmentation_flag)
            # 更新类的训练集以及验证集
            solver.train_loader, solver.val_loader = train_loader_, val_loader_
            
            # 针对不同mode，在第二阶段的处理方式
            if config.mode == 'train' or config.mode == 'train_stage2':
                solver.train_stage2(index)
            else:
                best_thr, best_pixel_thr, score = solver.choose_threshold(os.path.join(config.save_path, '%s_%d_%d_best.pth' % (config.model_type, 2, index)), index)
                scores.append(score)
                best_thrs.append(best_thr)
                best_pixel_thrs.append(best_pixel_thr)
                result[str(index)] = [best_thr, best_pixel_thr, score]

    # 若为选阈值操作，则输出n_fold折验证集结果的平均值
    if config.mode == 'choose_threshold':
        score_mean = np.array(scores).mean()
        thr_mean = np.array(best_thrs).mean()
        pixel_thr_mean = np.array(best_pixel_thrs).mean()
        print('score_mean:{}, thr_mean:{}, pixel_thr_mean:{}'.format(score_mean, thr_mean, pixel_thr_mean))
        result['mean'] = [float(thr_mean), float(pixel_thr_mean), float(score_mean)]

        with codecs.open(config.save_path + '/result.json', 'w', "utf-8") as json_file:
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
        stage set，注意若当two_stage等于False的时候，epoch_stage2必须等于0，否则会影响到学习率衰减。其余参数以stage1的配置为准
        当save_step为10时，epoch_stage1和epoch_stage2必须是10的整数
        当前的resume在第一阶段只考虑已经训练了超过epoch_stage1_freeze的情况，当mode=traim_stage2时，resume必须有值
        
        若第一阶段和第二阶段均训练，则two_stage为true，并设置相应的epoch_stage1、epoch_stage2，mode设置为train
        若只训练第一阶段，则two_stage改为False，并设置epoch_stage2=0，mode设置为train
        若只训练第二阶段，则two_stage为true，并设置相应的resume权重名称，mode设置为train_stage2
        训练过程中断了，则设置相应的resume权重名称
        
        若选第一阶段的阈值，则mode设置为choose_threshold，two_stage设置为False
        若选第二阶段的阈值，则mode设置为choose_threshold，two_stage设置为True

        第一阶段为768，第二阶段为1024，unet_resnet34时各个电脑可以设置的最大batch size
        zdaiot:10,6 z840:12,6 mxq:20,8
        '''
        parser.add_argument('--two_stage', type=bool, default=True, help='if true, use two_stage method')
        parser.add_argument('--image_size_stage1', type=int, default=768, help='image size in the first stage')
        parser.add_argument('--batch_size_stage1', type=int, default=20, help='batch size in the first stage')
        parser.add_argument('--epoch_stage1', type=int, default=45, help='How many epoch in the first stage')
        parser.add_argument('--epoch_stage1_freeze', type=int, default=0, help='How many epoch freezes the encoder layer in the first stage')

        parser.add_argument('--image_size_stage2', type=int, default=1024, help='image size in the second stage')
        parser.add_argument('--batch_size_stage2', type=int, default=8, help='batch size in the second stage')
        parser.add_argument('--epoch_stage2', type=int, default=20, help='How many epoch in the second stage')
        parser.add_argument('--epoch_stage2_accumulation', type=int, default=0, help='How many epoch gradients accumulate in the second stage')
        parser.add_argument('--accumulation_steps', type=int, default=10, help='How many steps do you add up to the gradient in the second stage')

        parser.add_argument('--augmentation_flag', type=bool, default=True, help='if true, use augmentation method in train set')
        parser.add_argument('--n_splits', type=int, default=5, help='n_splits_fold')

        # model set
        parser.add_argument('--resume', type=str, default=0, help='if has value, must be the name of Weight file.')
        parser.add_argument('--mode', type=str, default='choose_threshold', help='train/train_stage2/choose_threshold. if train_stage2, will train stage2 only and resume cannot empty')
        parser.add_argument('--model_type', type=str, default='unet_resnet34', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/unet_resnet34/linknet/deeplabv3plus/pspnet_resnet34')

        # model hyper-parameters
        parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
        parser.add_argument('--img_ch', type=int, default=3)
        parser.add_argument('--output_ch', type=int, default=1)
        parser.add_argument('--num_epochs_decay', type=int, default=70) # TODO
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--lr', type=float, default=0.0002, help='init lr in stage1')
        parser.add_argument('--lr_stage2', type=float, default=0.00001, help='init lr in stage2')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay in optimizer')
        
        # dataset 
        parser.add_argument('--model_path', type=str, default='./checkpoints')
        parser.add_argument('--dataset_root', type=str, default='./datasets/SIIM_data')
        parser.add_argument('--train_path', type=str, default='./datasets/SIIM_data/train_images')
        parser.add_argument('--mask_path', type=str, default='./datasets/SIIM_data/train_mask')

        config = parser.parse_args()
        # config = {k: v for k, v in args._get_kwargs()}
    if config.two_stage == False and config.mode != 'choose_threshold':
        assert config.epoch_stage2 == 0,'当two_stage等于False的时候，epoch_stage2必须等于0，否则会影响到学习率衰减'
    if config.mode == 'train_stage2':
        assert config.resume != ''
    main(config)
