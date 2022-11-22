# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 训练
'''
import os
import time
import copy
import torch
import random
import numpy as np
import os.path as osp
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler  # need pytorch>1.6
import matplotlib.pyplot as plt
from utils.utils import AverageMeter, init_logger, exp_smoothing
from utils.metric import Metric

from utils.losses import JointLoss
from utils.optimzer import build_optimizer
from utils.lr_scheduler import PolyScheduler
from datasets.s2looking import s2looking_dataset, get_train_transform, get_val_transform
from models import create_model
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, SoftBCEWithLogitsLoss


def train(CFG):
    # 参数
    D = CFG['dataset']
    class_info = D['cls_info']
    n_class = len(class_info.items())
    train_dirs = D['train_dirs']
    val_dirs = D['val_dirs']
    resample = D['resample']
    ignore_index = D['ignore_index']

    T = CFG['train']
    max_epochs = T['epochs']
    batch_size = T['batch_size']
    smoothing = T['smoothing']
    save_inter = T['save_inter']
    log_inter = T['log_inter']
    plot = T['plot']
    log_dir = os.path.join(CFG['run_dir'], CFG['run_name'])
    ckpt_dir = os.path.join(log_dir, 'ckpt')

    cfgN = CFG['network']
    cfgOptim = CFG['optimizer']

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 准备数据集
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    train_data = s2looking_dataset(train_dirs, train_transform, mode='train')
    val_data = s2looking_dataset(val_dirs, val_transform, mode='val')
    train_data_size = train_data.__len__()
    val_data_size = val_data.__len__()
    img_c, img_h, img_w = train_data.__getitem__(0)['image1'].shape
    assert img_c == 3, f'img_c:{img_c}'

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=batch_size,
        drop_last=True)
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=batch_size,
        drop_last=False)

    # logger
    logger = init_logger(os.path.join(log_dir, time.strftime("%m-%d-%H-%M-%S", time.localtime()) + '.log'))
    logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'
                .format(max_epochs, img_w, img_h, train_data_size, val_data_size))

    # 网络
    model = create_model(cfg=cfgN).cuda()
    model = torch.nn.DataParallel(model)

    # 训练
    # optimizer = build_optimizer(model, cfgOptim)
    optimizer = optim.AdamW([{"params": [param for name, param in model.named_parameters()
                                             if "encoder" in name], "lr": cfgOptim['lr']},
                            {"params": [param for name, param in model.named_parameters()
                                             if "encoder" not in name], "lr": cfgOptim['lr'] * 10.0}],
                            lr=cfgOptim['lr'], weight_decay=cfgOptim['weight_decay'])
    # optimizer = optim.SGD([{"params": [param for name, param in model.named_parameters()
    #                                          if "encoder" in name], "lr": cfgOptim['lr']},
    #                         {"params": [param for name, param in model.named_parameters()
    #                                          if "encoder" not in name], "lr": cfgOptim['lr'] * 10.0}],
    #                         lr=cfgOptim['lr'], momentum=0.9, weight_decay=cfgOptim['weight_decay'])
    scheduler = PolyScheduler(optimizer, power=4, total_steps=max_epochs, min_lr=1e-6, last_epoch=-1)

    # FocalLoss_fn = FocalLoss(mode='multiclass', ignore_index=ignore_index)
    DiceLoss_fn = DiceLoss(mode='multiclass', ignore_index=ignore_index)
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=smoothing, ignore_index=ignore_index)
    criterion = JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn, first_weight=0.5, second_weight=0.5).cuda()
    # criterion = SoftCrossEntropyLoss(smooth_factor=smoothing, ignore_index=ignore_index).cuda()

    train_loss_total_epochs, val_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    best_loss = 10000
    epoch_start = 0

    # 主循环
    for epoch in range(epoch_start, max_epochs):
        t0 = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            optimizer.zero_grad()
            img1, img2, label = batch_samples['image1'], batch_samples['image2'], batch_samples['label']
            img1, img2, label = img1.cuda(), img2.cuda(), label.squeeze(1).cuda()
            pred = model(img1, img2)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx > 0 and batch_idx % log_inter == 0:
                spend_time = time.time() - t0
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx / train_loader_size * 100,
                    optimizer.param_groups[-1]['lr'],
                    train_iter_loss.avg, spend_time / (batch_idx + 1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()

        # 验证阶段
        model.eval()
        val_epoch_loss = AverageMeter()
        val_iter_loss = AverageMeter()
        M = Metric(num_class=n_class, binary=True)
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(val_loader):
                img1, img2, label = batch_samples['image1'], batch_samples['image2'], batch_samples['label']
                img1, img2, label = img1.cuda(), img2.cuda(), label.squeeze(1).cuda()
                pred = model(img1, img2)
                loss = criterion(pred, label)
                pred = torch.softmax(pred, dim=1).cpu().numpy()
                pred = np.argmax(pred, axis=1)
                M.add_batch(pred, label.cpu().numpy())

                image_loss = loss.item()
                val_epoch_loss.update(image_loss)
                val_iter_loss.update(image_loss)

            val_loss = val_iter_loss.avg
            scores = M.evaluate()
            # Ps = scores['class_precision']
            # Rs = scores['class_recall']
            # IoUs = scores['class_iou']
            # mIoU = scores['mean_iou']
            logger.info('[val] epoch:{} loss:{:.6f} precision:{:.2f} recall:{:.2f} iou:{:.2f}'.format(
                epoch, val_loss, scores['precision'], scores['recall'], scores['iou']))

        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        val_loss_total_epochs.append(val_epoch_loss.avg)
        epoch_lr.append(scheduler.get_last_lr()[-1])
        scheduler.step()

        # 保存模型
        if save_inter > 0 and epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
        # 保存最优模型
        if val_loss < best_loss:
            state = {'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_loss = val_loss
            logger.info('[save] Best Model saved at epoch:{}'.format(epoch))
        logger.info('==============================================================')
        # 保存最后模型
        if epoch == max_epochs-1:
            state = {'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(ckpt_dir, 'checkpoint-final.pth')
            torch.save(state, filename)
            logger.info('[save] Final Model saved at epoch:{}'.format(epoch))

    # 训练loss曲线
    if plot:
        x = [i for i in range(max_epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, exp_smoothing(train_loss_total_epochs, 0.6), label='train loss')
        ax.plot(x, exp_smoothing(val_loss_total_epochs, 0.6), label='val loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('CrossEntropy', fontsize=15)
        ax.set_title('train curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr, label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title('lr curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.savefig(osp.join(log_dir, 'plot.png'))
        plt.show()


