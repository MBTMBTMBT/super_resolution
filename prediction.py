import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from models import *
from my_utils import load_model
from dataset import CroppingDataset


class Predictor(object):
    def __init__(
            self,
            input_size: tuple,
            model: torch.nn.Module,
    ):
        self.input_size = input_size
        self.model = model
        self.model.eval()

    def predict(self, img_x: torch.Tensor) -> np.ndarray:
        assert len(img_x.shape) == 4
        assert img_x.shape[2] == self.input_size[0] and img_x.shape[3] == self.input_size[1]
        with torch.no_grad():
            prediction = self.model(img_x)
        prediction = np.transpose(prediction.detach().numpy(), (0, 2, 3, 1))
        prediction = np.clip(prediction, 0, 1)
        return prediction


if __name__ == '__main__':
    model = FSRCNN()
    model = load_model(
        model,
        r'E:\my_files\programmes\python\super_resolution_outputs\FSRCNN-0\saved_checkpoints\model_param_10.pkl',
        device=torch.device("cpu"),
    )
    predictor = Predictor(
        input_size=(240, 240),
        model=model,
    )
    val_dataset = CroppingDataset(
        dataset_dir=r'E:\my_files\programmes\python\super_resolution_images\fold0',
        x_size=(240, 240),
        y_size=(480, 480),
        resize_mode='crop_middle',
        rand_flip=False,
        crop_middle_scale=4,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=True,
        num_workers=0,
        batch_size=1,
    )
    for img_x, img_y in val_loader:
        out = predictor.predict(img_x)
        plt.figure(0)
        plt.title('img_x')
        img_x = np.squeeze(np.transpose(img_x.detach().numpy(), (0, 2, 3, 1)))
        plt.imshow(img_x)
        plt.figure(1)
        plt.title('img_y')
        img_y = np.squeeze(np.transpose(img_y.detach().numpy(), (0, 2, 3, 1)))
        plt.imshow(img_y)
        plt.figure(2)
        plt.title('out')
        plt.imshow(np.squeeze(out))
        plt.show()
