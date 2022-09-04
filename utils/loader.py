import torch
from pathlib import Path
import cv2


class MnistLoadTest(object):
    def __init__(self, data_path):
        data_path = Path(data_path)
        self.img_path_list = sorted(data_path.glob("img/*.pt"))
        self.grandtruth_path_list = sorted(data_path.glob("transed_pt/*.pt"))
        self.erase_path_list = sorted(data_path.glob("erase_pt/*.pt"))

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, data_id):
        img = torch.load(str(self.img_path_list[data_id]))
        gts = torch.load(str(self.grandtruth_path_list[data_id]))
        erases = torch.load(str(self.erase_path_list[data_id]))
        return {
            'img': img.type(torch.FloatTensor),
            'img_denoised': erases.type(torch.FloatTensor),
            'gt': gts.type(torch.FloatTensor)
        }
