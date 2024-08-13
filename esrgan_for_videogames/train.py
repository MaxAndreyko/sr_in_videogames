# flake8: noqa
import os.path as osp

import esrgan_for_videogames.archs
import esrgan_for_videogames.data
import esrgan_for_videogames.losses
import esrgan_for_videogames.models
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
