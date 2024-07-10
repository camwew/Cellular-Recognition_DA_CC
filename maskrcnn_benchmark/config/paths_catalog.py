# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from copy import deepcopy

class DatasetCatalog(object):
    DATA_DIR = "/home/jianan/"
    DATASETS = {
        "coco_PanNuke_train_breast": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast/images",
            "ann_file": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast/instances_PanNuke_train.json",
            "inpaint_img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast/inpaint_images"
        },
        "coco_PanNuke_train_colon": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_colon/images",
            "ann_file": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_colon/instances_PanNuke_train.json",
            "inpaint_img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_colon/inpaint_images"
        },
        "coco_PanNuke_train_bile": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_bile/images",
            "ann_file": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_bile/instances_PanNuke_train.json",
            "inpaint_img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_bile/inpaint_images"
        },
        "coco_PanNuke_train_fakebreast_FDA": {
            "img_dir": "InstanceSeg+Class/main_code/FDA/PanNuke/colon2breast/images",
            "ann_file": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_colon/instances_PanNuke_train.json",
            "inpaint_img_dir": "InstanceSeg+Class/main_code/FDA/PanNuke/colon2breast/inpaint_images"
        },
        "coco_PanNuke_train_fakecolon_FDA": {
            "img_dir": "InstanceSeg+Class/main_code/FDA/PanNuke/breast2colon/images",
            "ann_file": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast/instances_PanNuke_train.json",
            "inpaint_img_dir": "InstanceSeg+Class/main_code/FDA/PanNuke/breast2colon/inpaint_images"
        },
        "coco_PanNuke_train_fakebile_from_colon_FDA": {
            "img_dir": "InstanceSeg+Class/main_code/FDA/PanNuke/colon2bile/images",
            "ann_file": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_colon/instances_PanNuke_train.json",
            "inpaint_img_dir": "InstanceSeg+Class/main_code/FDA/PanNuke/colon2bile/inpaint_images"
        },
        "coco_PanNuke_train_breast_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_fakebreast_FDA/fold3/train/0050400/instances_mrcn_predict.json",
            "inpaint_img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast/inpaint_images_mrcn_predict"
        },
        "coco_PanNuke_train_colon_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_colon/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_fakecolon_FDA/fold3/train/0064800/instances_mrcn_predict.json",
            "inpaint_img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_colon/inpaint_images_mrcn_predict"
        },
        "coco_PanNuke_train_thyroid_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_thyroid/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_thyroid/fold3/train/breast_0064800/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_thyroid/fold3/train/breast_0064800/inpaint_images"
        },
        "coco_PanNuke_train_skin_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_skin/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_skin/fold3/train/breast_0050400/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_skin/fold3/train/breast_0050400/inpaint_images"
        },
        "coco_PanNuke_train_bile_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_bile/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_bile/fold3/train/breast_0050400/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_bile/fold3/train/breast_0050400/inpaint_images"
        },
        "coco_PanNuke_train_testis_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_testis/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_testis/fold3/train/breast_0057600/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_testis/fold3/train/breast_0057600/inpaint_images"
        },
        "coco_PanNuke_train_prostate_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_prostate/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_prostate/fold3/train/breast_0050400/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_prostate/fold3/train/breast_0050400/inpaint_images"
        },
        "coco_PanNuke_train_breast_13_2": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast_13_2/images",
            "ann_file": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast_13_2/instances_PanNuke_train.json",
            "inpaint_img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast_13_2/inpaint_images"
        },
        "coco_PanNuke_train_breast_23_1": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast_23_1/images",
            "ann_file": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast_23_1/instances_PanNuke_train.json",
            "inpaint_img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_breast_23_1/inpaint_images"
        },
        "coco_PanNuke_train_thyroid_13_2_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_thyroid_13_2/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_thyroid/fold2/train/breast_0050400/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_thyroid/fold2/train/breast_0050400/inpaint_images"
        },
        "coco_PanNuke_train_thyroid_23_1_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_thyroid_23_1/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_thyroid/fold1/train/breast_0072000/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_thyroid/fold1/train/breast_0072000/inpaint_images"
        },
        "coco_PanNuke_train_bile_13_2_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_bile_13_2/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_bile/fold2/train/breast_0043200/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_bile/fold2/train/breast_0043200/inpaint_images"
        },
        "coco_PanNuke_train_bile_23_1_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_bile_23_1/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_bile/fold1/train/breast_0057600/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_bile/fold1/train/breast_0057600/inpaint_images"
        },
        "coco_PanNuke_train_testis_13_2_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_testis_13_2/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_testis/fold2/train/breast_0057600/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_testis/fold2/train/breast_0057600/inpaint_images"
        },
        "coco_PanNuke_train_testis_23_1_mrcn_predict": {
            "img_dir": "InstanceSeg+Class/raw_datasets/PanNuke/train/patches_testis_23_1/images",
            "ann_file": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_testis/fold1/train/breast_0043200/instances_mrcn_predict_ins_mask.json",
            "inpaint_img_dir": "InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_testis/fold1/train/breast_0043200/inpaint_images"
        },}
        

        

    

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                inpaint_root=os.path.join(data_dir, attrs["inpaint_img_dir"])
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = deepcopy(DatasetCatalog.DATASETS[name])
            attrs["img_dir"] = os.path.join(data_dir, attrs["img_dir"])
            attrs["ann_dir"] = os.path.join(data_dir, attrs["ann_dir"])
            return dict(factory="CityScapesDataset", args=attrs)

        elif "cell" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
                inpaint_root=os.path.join(data_dir, attrs[2])
            )
            return dict(
                factory="COCODataset",
                args=args,
            )

        elif "em" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
                #    sem_root = os.path.join(data_dir, attrs[2]),

            )

            return dict(
                factory="COCODataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
