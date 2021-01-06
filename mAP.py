import os
import sys
import mrcnn.utils as utils
import mextra.utils as extra_utils
import numpy as np
import mcoco.coco as coco
import mrcnn.model as modellib
from mrcnn.config import Config

ROOT_DIR = os.path.abspath(
    "/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/2-maskR-CNN-carlane")
DATA_DIR = os.path.join(ROOT_DIR, "data/data-road")  #
MODEL_DIR = os.path.join(ROOT_DIR, "logs/road20201211T2131")

dataset_test = coco.CocoDataset()
dataset_test.load_coco(DATA_DIR, subset="drivable_validate", year="2020")
dataset_test.prepare()


class InferenceConfig(Config):
    NAME = "road"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_DIM = 800
    IMAGE_RESIZE_MODE = "square"

    TRAIN_ROIS_PER_IMAGE = 128

    STEPS_PER_EPOCH = 500

    VALIDATION_STEPS = 50

    BACKBONE = "resnet50"

    POST_NMS_ROIS_INFERENCE = 512


config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

weights_path = os.path.join(MODEL_DIR, "mask_rcnn_road_0160.h5")

print("Loading weights", weights_path)
model.load_weights(weights_path, by_name=True)

print("______________________just wait, keep patient_________________________")

start = timen = predictions = \
    extra_utils.compute_multiple_per_class_precision(model, config, dataset_test, number_of_images=400,
                                                     iou_threshold=0.5)
complete_predictions = []

for drivable in predictions:
    complete_predictions += predictions[drivable]
    print("{}  ({}):  {}".format(drivable, len(predictions[drivable]), np.mean(predictions[drivable])))

print("----------------------------------------------------------------------")
print("average precision: {}".format(np.mean(complete_predictions)))
