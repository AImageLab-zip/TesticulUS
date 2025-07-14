
import torch, torchvision
from torch.utils.data import Dataset
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image


class USImages(Dataset):
    def __init__(self, root_path, transforms=None):
        assert Path(root_path).exists(), "provided path does not exist!!!"
        assert Path(root_path).joinpath("data").exists(), "./data folder does not exist!!!"

        self.root_path = root_path
        self.data_path = Path(self.root_path).joinpath("data")
        self.items = []
        for img_path in self.data_path.iterdir():
            self.items.append(f"{str(img_path).removesuffix('.bmp')}_0")
            self.items.append(f"{str(img_path).removesuffix('.bmp')}_1")
        self.transofrms = v2.Identity() if transforms is None else transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path = self.items[index][:-2] + ".bmp"
        side = self.items[index][-1:]
        img = Image.open(img_path)
        img = img.crop((185 + 300 * int(side), 130, 300 + 185 + 300 * int(side), 430))
        return self.transofrms(img)
