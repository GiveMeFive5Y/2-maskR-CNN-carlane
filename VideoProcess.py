from keras.preprocessing import image
import os
import matplotlib.pyplot as plt
from mrcnn.config import Config
import mrcnn.model as modellib
import cv2
from moviepy.editor import VideoFileClip
from mrcnn import visualize_drivable
import scipy
import skimage


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def process_video(image, title="", ax=None):
    results = model.detect([image], verbose=1)
    r = results[0]
    image1 = visualize_drivable.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,
                                                  r['scores'])
    # cv2.namedWindow("IMG",cv2.WINDOW_NORMAL)
    # cv2.imshow("IMG",image1)
    # cv2.waitKey()
    # cv2.destroyWindows()
    return image1


class InferenceConfig(Config):
    NAME = "road"

    BACKBONE = "resnet101"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1

    IMAGE_MIN_DIM = 1280
    IMAGE_MAX_DIM = 1920
    IMAGE_RESIZE_MODE = "square"

    STEPS_PER_EPOCH = 500
    TRAIN_ROIS_PER_IMAGE = 128
    VALIDATION_STEPS = 50
    POST_NMS_ROIS_INFERENCE = 512


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    HOME_DIR = os.path.abspath(
        "/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/2-maskR-CNN-carlane")
    MODEL_DIR = os.path.join(HOME_DIR, "logs/road20201211T2131")
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_road_0160.h5")

    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

    print("-----------The model is loading----------")

    model.load_weights(model_path, by_name=True)

    class_names = ['BG', "road"]

    # image_test = process_video(image_path)
    # scipy.misc.imsave(os.path.join(HOME_DIR, "TestImage.png"), image_test)

    output = os.path.join(HOME_DIR, "VideoOutput-16-2131-2.mp4")
    # clip1 = VideoFileClip(os.path.join("/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3"
    #                                    "/A-code/cabc30fc-e7726578.mov"))
    clip1 = VideoFileClip(os.path.join(HOME_DIR,"TestVideo.mp4"))
    clip = clip1.fl_image(process_video)
    clip.write_videofile(output, audio=False)

    print("Process Successfully")
