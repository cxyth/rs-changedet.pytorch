# -*- coding: utf-8 -*-
import cv2
from tqdm import tqdm
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.s2looking import s2looking_dataset, get_train_transform, get_val_transform
from models import create_model


def inference(CFG):
    cfgD = CFG['dataset']
    class_info = cfgD['cls_info']
    n_class = len(class_info.items())
    cfgN = CFG['network']
    cfgE = CFG['eval']
    ckpt = os.path.join(CFG['run_dir'], CFG['run_name'], "ckpt", cfgE['ckpt_name'])
    input_dir = cfgE['test_dir']
    base_dir = os.path.join(CFG['run_dir'], CFG['run_name'], cfgE['save_dir'])
    batch_size = cfgE['batch_size']
    save_dir = os.path.join(base_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)

    # 网络
    model = create_model(cfg=cfgN).cuda()

    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    val_transform = get_val_transform()
    testset = s2looking_dataset(dataset_url=input_dir, transform=val_transform, mode='test')
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    tbar = tqdm(testloader)
    with torch.no_grad():
        for batch_samples in tbar:
            img1, img2, fnames = batch_samples['image1'].cuda(), batch_samples['image2'].cuda(), batch_samples['fname']
            outputs = model(img1, img2)
            change_maps = torch.softmax(outputs, dim=1).cpu().numpy()

            #post processing
            change_masks = np.argmax(change_maps, axis=1).astype(np.uint8)

            for i in range(change_masks.shape[0]):
                fname = fnames[i]
                _mask = change_masks[i].squeeze()
                cv2.imwrite(os.path.join(save_dir, fname), _mask * 255)

