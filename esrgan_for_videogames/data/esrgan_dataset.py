import os
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SRDataset(data.Dataset):
    """Example dataset.

    1. Read GT image
    2. Generate LQ (Low Quality) image with cv2 bicubic downsampling and JPEG compression

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.root_folder = opt['dataroot']
        self.hq_folder = os.path.join(self.root_folder, opt['hq_folder'])
        self.lq_folder = os.path.join(self.root_folder, opt['lq_folder'])

        with open(os.path.join(self.root_folder, opt["txt_name"] + ".txt"), "r") as file:
            self.val_img_list = file.read().split("\n")

        # it now only supports folder mode, for other modes such as lmdb and meta_info file, please see:
        # https://github.com/xinntao/BasicSR/blob/master/basicsr/data/
        self.hq_paths = list(map(lambda x: os.path.join(self.hq_folder, x), self.val_img_list))
        self.lq_paths = list(map(lambda x: os.path.join(self.lq_folder, x), self.val_img_list))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        # Load gt images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        hq_path = self.hq_paths[index]
        img_hq_bytes = self.file_client.get(hq_path, 'hr')
        img_hq = imfrombytes(img_hq_bytes, float32=True)

        lq_path = self.lq_paths[index]
        img_lq_bytes = self.file_client.get(lq_path, 'lr')
        img_lq = imfrombytes(img_lq_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['hq_size']
            # random crop
            img_hq, img_lq = paired_random_crop(img_hq, img_lq, gt_size, scale)
            # flip, rotation
            img_hq, img_lq = augment([img_hq, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_hq, img_lq = img2tensor([img_hq, img_lq], bgr2rgb=True, float32=True)

        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_hq, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_hq, 'lq_path': lq_path, 'gt_path': hq_path}

    def __len__(self):
        return len(self.hq_paths)

