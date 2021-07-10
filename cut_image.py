from option import args
import cv2
from tqdm import tqdm
import os
import pickle
from pathlib import Path
import numpy as np


def cut_img(img_path_list, save_path, stride):
    os.makedirs(f'{save_path}256', exist_ok=True)
    num = 0
    for path in tqdm(img_path_list):
        img = cv2.imread(str(path))
        for top in range(0, img.shape[0], stride):
            for left in range(0, img.shape[1], stride):
                piece = np.zeros([256, 256, 3], np.uint8)
                temp = img[top:top+256, left:left+256, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                with open(f'{save_path}256/{num}.pkl', 'wb') as f:
                    pickle.dump(piece, f)
                num += 1

if __name__ == '__main__':
    p = Path(args.dir_data)
    train_input_files = [p_img for p_img in (p/'train_input_img').glob('*.png')]
    train_label_files = [p_img for p_img in (p/'train_label_img').glob('*.png')]
    cut_img(train_input_files, f'{args.dir_data}/train_input_img_', args.patch_size)
    cut_img(train_label_files, f'{args.dir_data}/train_label_img_', args.patch_size)