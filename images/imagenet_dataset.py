import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImgaeNetDataset(Dataset):
    def __init__(self, img_dir, img_size, transforms_list = []):
        self.img_dir = img_dir
        self.img_size = img_size # (256,256)
        self.file_name = []
        self.labels = []
        for c in os.listdir(os.path.join(self.img_dir)):
            for name in os.listdir(os.path.join(self.img_dir, c)):
                self.file_name.append(os.path.join(c, name))
                self.labels.append(int(c))
        self.loader = transforms.Compose([transforms.ToTensor(), *transforms_list])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_name[idx])
        image = Image.open(img_path).convert('RGB')
        # crop image to have same width & height
        w, h = image.size
        if w>h:
            offset = (int)((w - h)/2)
            image = image.crop((offset, 0, offset+h, h))
        elif h>w:
            offset = (int)((h - w)/2)
            image = image.crop((0, offset, w, offset+w))
        image = image.resize(self.img_size)
        image = self.loader(image)
        image = image * 2 - 1 # [0,1]->[-1,1]

        label = self.labels[idx]

        return image, label
