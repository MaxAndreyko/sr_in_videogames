import yaml
import logging
import os
import random

from basicsr.utils import scandir


def main(opt):
    logger = logging.getLogger(main.__name__)
    root_folder = opt['dataroot']
    hq_folder = os.path.join(root_folder, opt['hq_folder'])
    lq_folder = os.path.join(root_folder, opt['lq_folder'])
    logger.info("Getting LQ and HQ images paths ...")
    hq_img_names = [os.path.basename(os.path.join(hq_folder, v)) for v in list(scandir(hq_folder))]
    lq_img_names = [os.path.basename(os.path.join(lq_folder, v)) for v in list(scandir(lq_folder))]
    assert len(hq_img_names) == len(lq_img_names), "Number of images in LQ and HQ folders doesn't match"
    logger.info(f"Found {len(lq_img_names)} images in LQ and HQ folders")

    logger.info("Splitting data into train and validation samples ...")
    random.seed(opt["random_state"])
    random.shuffle(lq_img_names)
    split_ind = int(len(lq_img_names) * opt["ratio"])
    img_names_train = lq_img_names[split_ind:]
    img_names_val = lq_img_names[:split_ind]

    with open(os.path.join(root_folder, opt['train_txt_name'] + ".txt"), "w") as file:
        file.write("\n".join(img_names_train))
    with open(os.path.join(root_folder, opt['val_txt_name']  + ".txt"), "w") as file:
        file.write("\n".join(img_names_val))
    logger.info(f"Text files with split information are saved into: {root_folder}")


if __name__ == '__main__':
    with open("./options/esrgan_option.yml", "r") as file:
        opt = yaml.safe_load(file)
    main(opt["preparation"]["split"])