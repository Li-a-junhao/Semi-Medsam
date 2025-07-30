import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import json
from PIL import Image
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Throatdataset(Dataset):
    def __init__(self, args, base_dir=None, index=None, mode='train', aug_type="weak", Normalized=True,
                 resize_shape=None):
        self.args = args
        self._base_dir = base_dir
        self.aug_type = aug_type
        self.anns = json.load(open(base_dir, 'r'))

        self.dict = self.anns[mode]
        self.dict_use = []
        for i in range(len(index)):
            self.dict_use.append(self.dict[index[i]])
        self.index = index

        self.Normalized = Normalized
        self.image_net_mean = [0.485, 0.456, 0.406]
        self.image_net_std = [0.229, 0.224, 0.225]
        self.normalize_imagenet = transforms.Compose([transforms.Normalize(self.image_net_mean, self.image_net_std)])

        self.img_transform_w = transforms.Compose([
            transforms.Resize((resize_shape, resize_shape)),
            ToTensor_img()])

        self.gt_transform_w = transforms.Compose([
            transforms.Resize((args.label_shape, args.label_shape), interpolation=InterpolationMode.NEAREST),
            ToTensor_gt()])

        self.img_transform_s = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.9, 1.1)),
            transforms.GaussianBlur(5)])

        self.gt_transform_s = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.9, 1.1)),
        ])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        root = self._base_dir.split("/", 6)[0:-1]
        root = "/".join(root) + "/"
        image_item = self.dict_use[index]
        image_path = image_item['image_path']
        label_path = os.path.join(root, image_item['label_path'])
        NBI_image_path = image_item['mode_nbi_image_path']
        NBI_label_path = image_item['mode_nbi_label_path']
        img_w = Image.open(image_path).convert('RGB')
        img_n = Image.open(NBI_image_path).convert('RGB')
        label_w = Image.open(label_path).convert('L')
        label_n = Image.open(NBI_label_path).convert('L')

        assert len(np.array(img_w)) == len(np.array(label_w)) and len(np.array(img_w)[0]) == len(np.array(label_w)[0])
        assert len(np.array(img_n)) == len(np.array(label_n)) and len(np.array(img_n)[0]) == len(np.array(label_n)[0])

        seed = np.random.randint(2147483647)
        if self.aug_type == "weak":
            img_w = img_w
            img_n = img_n
            label_w = label_w
            label_n = label_n

        if self.aug_type == "strong":
            torch.manual_seed(seed)
            img_w = self.img_transform_s(img_w)
            torch.manual_seed(seed)
            label_w = self.gt_transform_s(label_w)
            torch.manual_seed(seed)
            img_n = self.img_transform_s(img_n)
            torch.manual_seed(seed)
            label_n = self.gt_transform_s(label_n)

        imgw_arr = np.array(img_w)
        index = np.where((imgw_arr[:, :, 0] == 0) & (imgw_arr[:, :, 1] == 0) & (imgw_arr[:, :, 2] == 0))
        labelw_arr = np.array(label_w)
        labelw_arr = labelw_arr / 255
        labelw_arr[index] = 255
        label_w = Image.fromarray(labelw_arr)

        img_w, img_n = self.img_transform_w(img_w), self.img_transform_w(img_n)
        label_w = self.gt_transform_w(label_w)

        img_w_normalized = self.normalize_imagenet(img_w.permute(2, 0, 1) / 255).permute(1, 2, 0)
        img_n_normalized = self.normalize_imagenet(img_n.permute(2, 0, 1) / 255).permute(1, 2, 0)
        sample = {"img_w": img_w, "label_w": label_w, "img_n": img_n, "img_w_norm": img_w_normalized, "img_n_norm": img_n_normalized}
        return sample, image_item


class ToTensor_gt(object):
    def __call__(self, label):
        label = np.array(label)
        return (torch.from_numpy(label)).long()


class ToTensor_img(object):
    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        return torch.from_numpy(img)
