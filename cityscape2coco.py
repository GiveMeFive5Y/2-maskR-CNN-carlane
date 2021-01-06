import cv2
import json
import os
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

ROOT_DIR = os.getcwd()
IMAGE_DIR = os.path.join(ROOT_DIR,"data/leftImg8bit/")
ANNOTATION_DIR = os.path.join(ROOT_DIR,"data/gtFine/")
ANNOTATION_SAVE_DIR = os.path.join(ROOT_DIR,"data/annotations-15/")
INSTANCE_DIR = os.path.join(ROOT_DIR,"data/instances-15/")
IMAGE_SAVE_DIR = os.path.join(ROOT_DIR,"data/val_images-15/")


INFO = {
    "description": "Cityscapes_Instance Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": "2020",
    "contributor": "Xavier",
    "date_created": "2020-09-10 16:16:16.123456"
}
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]
CATEGORIES = [
    {
        'id': 1,
        'name': 'road',
        'supercategory': 'cityscapes'
    },
    {
        'id': 2,
        'name': 'sidewalk',
        'supercategory': 'cityscapes'
    },
    {
        'id': 3,
        'name': 'traffic_light',
        'supercategory': 'cityscapes'
    },
    {
        'id': 4,
        'name': 'traffic_sign',
        'supercategory': 'cityscapes'
    },
    {
        'id': 5,
        'name': 'person',
        'supercategory': 'cityscapes'
    },
    {
        'id': 6,
        'name': 'rider',
        'supercategory': 'cityscapes'
    },
    {
        'id': 7,
        'name': 'car',
        'supercategory': 'cityscapes'
    },
    {
        'id': 8,
        'name': 'truck',
        'supercategory': 'cityscapes'
    },
    {
        'id': 9,
        'name': 'bus',
        'supercategory': 'cityscapes'
    },
    {
        'id': 10,
        'name': 'caravan',
        'supercategory': 'cityscapes'
    },
    {
        'id': 11,
        'name': 'trailer',
        'supercategory': 'cityscapes'
    },
    {
        'id': 12,
        'name': 'train',
        'supercategory': 'cityscapes'
    },
    {
        'id': 13,
        'name': 'motorcycle',
        'supercategory': 'cityscapes'
    },
    {
        'id': 14,
        'name': 'bicycle',
        'supercategory': 'cityscapes'
    },
    {
        'id': 15,
        'name': 'building',
        'supercategory': 'cityscapes'
    },
]

background_label = list(range(-1, 7, 1)) + list(range(9, 11, 1)) + list(range(12,19,1)) + list(range(21,24,1))
idx = 0
pic_scale = 1.0
h_bias = 1.0

#将InstanceTrainId文件保存到annotations文件夹下,并将原图保存至val_images文件夹中

def image_trans():
    img_subfolders = os.listdir(IMAGE_DIR)
    image_count = 0
    for sub in img_subfolders:
        image_sub_path = os.path.join(IMAGE_DIR, sub)
        for image in os.listdir(image_sub_path):
            img_path = os.path.join(image_sub_path, image)
            ann_name = image.split('_')[0] + '_' + image.split('_')[1] + '_' + image.split('_')[2] + '_gtFine_instanceTrainIds15.png'
            ann_sub_path = os.path.join(ANNOTATION_DIR, sub)
            ann_path = os.path.join(ann_sub_path, ann_name)
            if os.path.exists(ann_path):
                pic = cv2.imread(img_path)
                h, w = pic.shape[:2]
                new_w = w * pic_scale
                new_h = new_w / 2
                top = int((h_bias*h-new_h)/2)
                bottom = int((h_bias*h+new_h)/2)
                left = int((w-new_w)/2)
                right = int((w+new_w)/2)
                roi = pic[top:bottom, left:right]
                img_save_path = os.path.join(IMAGE_SAVE_DIR, image)
                cv2.imwrite(img_save_path, roi)
                annotation = cv2.imread(ann_path, -1)
                ann_roi = annotation[top:bottom, left:right]
                ann_save_path = os.path.join(ANNOTATION_SAVE_DIR, ann_name)
                cv2.imwrite(ann_save_path, ann_roi)
            else:
                print(image + '  do not have instance annotation')
            print(image_count)
            image_count += 1

#生成instance_dir文件夹，每个文件夹中包括了每张图片的labelTrainId图片
def data_loader():
    imgs = os.listdir(IMAGE_SAVE_DIR)
    masks_generator(imgs, ANNOTATION_DIR)
    # masks_generator(imgs, os.path.join(ANNOTATION_DIR,"train/"))


def masks_generator(imges, ann_path):
    global idx
    pic_count = 0
    for pic_name in imges:
        image_name = pic_name.split('.')[0]
        ann_folder = os.path.join(INSTANCE_DIR, image_name)
        os.mkdir(ann_folder)
        annotation_name = pic_name.split('_')[0] + '_' + pic_name.split('_')[1] + '_' + pic_name.split('_')[2] + '_gtFine_instanceIds.png'
        # annotation_name = image_name + '_instanceIds.png'
        annotation = cv2.imread(os.path.join(ann_path, annotation_name), -1)
        h, w = annotation.shape[:2]
        ids = np.unique(annotation)
        for id in ids:
            if id in background_label:
                continue
            else:
                class_id = id
                if class_id == 7:
                    instance_class = 'road'
                elif class_id == 8:
                    instance_class = 'sidewalk'
                elif class_id == 11:
                    instance_class = 'building'
                elif class_id == 19:
                    instance_class = 'traffic light'
                elif class_id == 20:
                    instance_class = 'traffic sign'
                elif class_id == 24:
                    instance_class = 'person'
                elif class_id == 25:
                    instance_class = 'rider'
                elif class_id == 26:
                    instance_class = 'car'
                elif class_id == 27:
                    instance_class = 'truck'
                elif class_id == 28:
                    instance_class = 'bus'
                elif class_id == 29:
                    instance_class = 'caravan'
                elif class_id == 30:
                    instance_class = 'trailer'
                elif class_id == 31:
                    instance_class = 'train'
                elif class_id == 32:
                    instance_class = 'motorcucle'
                elif class_id == 33:
                    instance_class = 'bicycle'
                else:
                    continue
            instance_mask = np.zeros((h, w, 3), dtype=np.uint8)
            mask = annotation == id
            instance_mask[mask] = 255
            mask_name = image_name + '_' + instance_class + '_' + str(idx) + '.jpg'
            print("mask_name",mask_name)
            cv2.imwrite(os.path.join(ann_folder, mask_name), instance_mask)
            idx += 1
        pic_count += 1
        print("pic_count",pic_count)


def json_generate():
    road = 0; sidework  = 0; traffic_light = 0; traffic_sign = 0; person = 0; rider = 0;
    car = 0; truck = 0; bus = 0; caravan = 0; trailer = 0; train = 0; motorcycle =0; bicycle = 0; building = 0
    # files = os.listdir(IMAGE_SAVE_DIR)
    files = os.listdir(os.path.join(ROOT_DIR,"data","leftImg8bit", "val_images/"))  # train_images  OR val_images

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # go through each image
    for image_filename in files:
        image_name = image_filename.split('.')[0]
        # image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        image_path = os.path.join(os.path.join(ROOT_DIR, "data","leftImg8bit","val_images/"),image_filename) # train_images  OR val_images
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)
        print(image_filename)
        annotation_sub_path = os.path.join(INSTANCE_DIR, image_name)
        ann_files = os.listdir(annotation_sub_path)
        if len(ann_files) == 0:
            print("no avaliable annotation")
            continue
        else:
            for annotation_filename in ann_files:
                annotation_path = os.path.join(annotation_sub_path, annotation_filename)
                for x in CATEGORIES:
                    if x['name'] in annotation_filename:
                        class_id = x['id']
                        break
                # class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                if class_id == 1:
                    road += 1
                elif class_id == 2:
                    sidework += 1
                elif class_id == 3:
                    traffic_light += 1
                elif class_id == 4:
                    traffic_sign += 1
                elif class_id == 5:
                    person += 1
                elif class_id == 6:
                    rider += 1
                elif class_id == 7:
                    car += 1
                elif class_id == 8:
                    truck += 1
                elif class_id == 9:
                    bus += 1
                elif class_id == 10:
                    caravan += 1
                elif class_id == 11:
                    trailer += 1
                elif class_id == 12:
                    train += 1
                elif class_id == 13:
                    motorcycle += 1
                elif class_id == 14:
                    bicycle += 1
                elif class_id == 15:
                    building += 1
                else:
                    print('illegal class id')
                category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                binary_mask = np.asarray(Image.open(annotation_path)
                                         .convert('1')).astype(np.uint8)

                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)

                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1
            print(image_id)

    with open('{}/val_image-15.json'.format(ROOT_DIR), 'w') as output_json_file:  # train_images  OR val_images
        json.dump(coco_output, output_json_file)
    print(road)



if __name__ == "__main__":
    # image_trans()
    # data_loader()
    json_generate()