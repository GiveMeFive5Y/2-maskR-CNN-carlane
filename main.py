import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import mcoco.coco as coco

ROOT_DIR = os.path.abspath(
    "/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/2-maskR-CNN-carlane")

MODEL_PATH = os.path.join(ROOT_DIR, "model")

COCO_MODEL_PATH = os.path.join(MODEL_PATH, "mask_rcnn_drivable_res101.h5")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class ShapeConfig(Config):
    NAME = "road"

    BACKBONE = "resnet101"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1  # background + drivable

    IMAGES_MAX_DIM = 1024
    IMAGES_MIX_DIM = 800
    IMAGE_RESIZE_MODE = "square"

    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 1000

    POST_NMS_ROIS_INFERENCE = 512


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = ShapeConfig()

    DATA_DIR = ROOT_DIR + '/data/data-road'
    dataset_train = coco.CocoDataset()
    dataset_train.load_coco(DATA_DIR, subset="drivable_train", year="2020")
    dataset_train.prepare()

    dataset_validate = coco.CocoDataset()
    dataset_validate.load_coco(DATA_DIR, subset="drivable_validate", year="2020")
    dataset_validate.prepare()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox",
                                                               "mrcnn_mask"])
    model.train(dataset_train, dataset_validate, learning_rate=config.LEARNING_RATE, epochs=40, layers='heads')

    model.train(dataset_train, dataset_validate, learning_rate=config.LEARNING_RATE, epochs=120, layers="4+")

    model.train(dataset_train, dataset_validate, learning_rate=config.LEARNING_RATE / 10, epochs=160, layers="all")
