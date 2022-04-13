import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import operator
import torchvision.transforms.transforms as transforms
import random
import tqdm


class CroppingDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            x_size: tuple,
            y_size: tuple,
            resize_mode: str,
            rand_flip: bool,
            max_crop_scale=8,
            crop_middle_scale=4,
    ):
        """
        initialize the MultiClassDataset
        :param dataset_dir: the root directory of the dataset,
            inside it should contain several folders, with the names of the classes.
        :param x_size: the expected output size of the image
        :param resize_mode: select from 'random_crop', 'random_scale_crop', 'crop_middle'.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.x_size = x_size
        self.y_size = y_size
        self.resize_mode = resize_mode
        self.rand_flip = rand_flip
        self.max_crop_scale = max_crop_scale
        self.crop_middle_scale = crop_middle_scale

        # prepare lists for images
        self.img_path_list = list()
        for each_img in os.listdir(self.dataset_dir):
            if each_img.split('.')[-1] == 'jpg' \
                    or each_img.split('.')[-1] == 'JPG' \
                    or each_img.split('.')[-1] == 'png' \
                    or each_img.split('.')[-1] == 'PNG' \
                    or each_img.split('.')[-1] == 'jpeg':  # or each_img.split('.')[-1] == 'webp':
                img_path = os.path.join(self.dataset_dir, each_img)
                self.img_path_list.append(img_path)
            else:
                print("Exception: ", each_img)

        self.crop_y = transforms.RandomCrop(y_size, pad_if_needed=True)
        self.crop_y_scale = {
            1: self.crop_y,
        }
        for i in range(2, self.max_crop_scale + 1):
            self.crop_y_scale[i] = transforms.RandomCrop((y_size[0] * i, y_size[1] * i), pad_if_needed=True)
        self.resize_y = transforms.Resize(y_size, antialias=True)
        self.resize_x = transforms.Resize(x_size, antialias=True)
        self.flip = transforms.RandomHorizontalFlip()
        self.crop_middle \
            = transforms.CenterCrop((x_size[0] * self.crop_middle_scale, x_size[1] * self.crop_middle_scale))

    def join(self, dataset):
        """
        merge two datasets
        :param dataset: another Dataset instance
        :return: None
        """
        assert operator.eq(self.x_size, dataset.x_size)
        self.img_path_list += dataset.img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        # print('begin reading')
        img, success = CroppingDataset._read_image(img_path)
        if not success:
            print('Image error: ', img_path)
        # print('end reading')

        # change values into 0 ~ 1
        img = img.type(torch.float32)
        img /= 255

        # print('begin crop')

        if self.rand_flip:
            img = self.flip(img)

        img_y = None
        if self.resize_mode == 'random_crop':
            img_y = self.crop_y(img)
        elif self.resize_mode == 'random_scale_crop':
            count = 0
            while True:
                scale = random.randint(1, self.max_crop_scale + 1)
                if self.y_size[0] * scale <= img.shape[1] and self.y_size[1] * scale <= img.shape[2]:
                    break
                elif count >= 3:
                    scale = 1
                    break
                else:
                    count += 1
            img_y = self.crop_y_scale[scale](img)
            img_y = self.resize_y(img_y)
        elif self.resize_mode == 'crop_middle':
            if self.y_size[0] * self.crop_middle_scale > img.shape[1] \
                    or self.y_size[1] * self.crop_middle_scale > img.shape[2]:
                img_y = self.resize_y(img)
            else:
                img_y = self.crop_middle(img)
                img_y = self.resize_y(img_y)
        img_x = self.resize_x(img_y)
        # print('end')

        return img_x, img_y

    @staticmethod
    def _read_image(img_path: str) -> (torch.Tensor, bool):
        """
        Args:
            img_path (str): Full path of the image.

        Returns:
            torch Tensor of the image
        """
        img, success = CroppingDataset.__read_image(img_path)
        img = torch.from_numpy(img)
        return img, success

    @staticmethod
    def __read_image(img_path: str) -> (np.ndarray, bool):
        img = cv2.imread(img_path)
        if img is None:
            img = cv2.imread(img_path)  # try again
        if img is None:
            img = np.zeros((3, 4096, 4096), dtype=np.uint8)
            return img, False
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # PIL sometimes can get up-side-down images, I hate it, so, lets use OpenCV~
        # img = Image.open(img_path)
        # img = f.convert('RGB')
        # img = np.array(img)
        return img.transpose((2, 0, 1)), True


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    test_dataset = CroppingDataset(
        dataset_dir=r'E:\my_files\programmes\python\super_resolution_images\fold0',
        x_size=(240, 240),
        y_size=(480, 480),
        resize_mode='random_scale_crop',
        rand_flip=True,
    )
    test_loader = DataLoader(test_dataset, shuffle=True, num_workers=2, batch_size=1, prefetch_factor=2)
    for imgs_x, imgs_y in tqdm.tqdm(test_loader):
        imgs_y = torch.squeeze(imgs_y)
        imgs_x = torch.squeeze(imgs_x)
        imgs_x = imgs_x.numpy()
        # print(imgs_y.shape)
        # print(imgs_x.shape)
        if len(imgs_x.shape) == 3:
            imgs_x = np.transpose(imgs_x, (1, 2, 0))
            plt.figure(1)
            plt.title('img x')
            plt.imshow(imgs_x)
            imgs_y = np.transpose(imgs_y, (1, 2, 0))
            plt.figure(2)
            plt.title('img y')
            plt.imshow(imgs_y)
            plt.show()

