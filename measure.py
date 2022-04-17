import numpy as np
import math

import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim
import matplotlib.pyplot as plt

from prediction import Predictor
from models import *
from my_utils import *
from dataset import *


def mse(slice_a: np.ndarray, slice_b: np.ndarray) -> float:
    return np.mean(np.square(np.subtract(slice_a, slice_b))).item()


def psnr(slice_a: np.ndarray, slice_b: np.ndarray):
    return 10 * math.log10(1 / mse(slice_a, slice_b))


def compare(img_a, img_b):
    mse_rst = mse(img_a, img_b)
    psnr_rst = psnr(img_a, img_b)
    ssim_rst = sk_ssim(img_a, img_b, channel_axis=2)
    return mse_rst, psnr_rst, ssim_rst


if __name__ == '__main__':
    model = FSRCNN()
    model = load_model(
        model,
        r'E:\my_files\programmes\python\super_resolution_outputs\FSRCNN-0\saved_checkpoints\model_param_59.pkl',
        device=torch.device("cpu"),
    )
    predictor = Predictor(
        # input_size=(240, 240),
        model=model,
    )
    val_dataset = ResizingDataset(
        dataset_dir=r'E:\my_files\programmes\python\super_resolution_images\srclassic\SR_testing_datasets\Set14',
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=True,
        num_workers=0,
        batch_size=1,
    )

    mse_lst, psnr_lst, ssim_lst = [], [], []
    count = 0
    for img_x, img_y in tqdm.tqdm(val_loader):
        out = predictor.predict(img_x)
        img_x = np.squeeze(np.transpose(img_x.detach().numpy(), (0, 2, 3, 1)))
        img_y = np.squeeze(np.transpose(img_y.detach().numpy(), (0, 2, 3, 1)))
        out = np.squeeze(out)
        mse_rst, psnr_rst, ssim_rst = compare(out, img_y)
        mse_lst.append(mse_rst)
        psnr_lst.append(psnr_rst)
        ssim_lst.append(ssim_rst)
        count += 1
        # plt.figure(0)
        # plt.title('img_x')
        # plt.imshow(img_x)
        # plt.figure(1)
        # plt.title('img_y')
        # plt.imshow(img_y)
        # plt.figure(2)
        # plt.title('out')
        # plt.imshow(np.squeeze(out))
        # plt.show()
        plt.imsave(r'E:\my_files\programmes\python\super_resolution_outputs\test_imgs\\' + str(count) + '.png', out)
    mse_mean = sum(mse_lst) / count
    psnr_mean = sum(psnr_lst) / count
    ssim_mean = sum(ssim_lst) / count
    print(mse_mean, psnr_mean, ssim_mean)
