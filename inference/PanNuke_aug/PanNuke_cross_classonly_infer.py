
import cv2
import os
import numpy as np
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import tifffile as tiff
import scipy.io as sio
import json
import torch
from metrics import mask2out, removeoverlap
from maskrcnn_benchmark.config import cfg
from inference.cell_predictor_classonly_multiclass import CellDemo
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.detector import build_detection_model



def infer_fluo2tcga(wts_root, test_root_name, test_root_json, out_pred_root, test_num = None):

    config_file = '../../configs/class_breast2thyroid_mrcn_predict.yaml'

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    cfg.merge_from_list(["MODEL.META_ARCHITECTURE", "GeneralizedRCNN_final"])
    print(cfg.MODEL.META_ARCHITECTURE)
    model = build_detection_model(cfg)

    cell_demo = CellDemo(
        cfg,
        min_image_size=1000,
        confidence_threshold=0.5,
        weight= wts_root,
        model=model
    )

    # put your testing images
    #test_root_name = ''


    # output saving root
    mkdir(os.path.join(out_pred_root, 'class'))
    #mkdir(os.path.join(out_pred_root, 'bbox'))
    #mkdir(os.path.join(out_pred_root, 'bi_mask'))
    mkdir(os.path.join(out_pred_root, 'ins_mask', 'binary'))
    for cat in range(1, 6):
        mkdir(os.path.join(out_pred_root, 'ins_mask', str(cat)))

    if test_num == None:
        test_imgs = os.listdir(test_root_name)
    else:
        test_imgs = os.listdir(test_root_name)[:test_num]
        
    fp = open(test_root_json, 'r', encoding='utf8')
    json_data = json.load(fp)
        
    for img_name in test_imgs:

        if img_name.endswith(".png"):
        
            for image_info in json_data['images']:
                if image_info['file_name'] == img_name:
                    img_idx = image_info['id']
                    break
            
            boxes = []
            for anno_info in json_data['annotations']:
                if anno_info['image_id'] == img_idx:
                    boxes.append(anno_info['bbox'])
            boxes = torch.as_tensor(boxes).reshape(-1, 4)
            target = BoxList(boxes, (256, 256), mode="xywh").convert("xyxy")
            target = target.clip_to_image(remove_empty=True)
            target = [target]
        
            image = cv2.imread(os.path.join(test_root_name, img_name))

            predictions, mask_list_binary, mask_list_allcat, box_centroids, labels_np = cell_demo.run_on_opencv_image(image, target)
            
            class_mat_name = os.path.splitext(img_name)[0] + '.mat'
            class_dict = {'inst_centroid': box_centroids, 'inst_type': labels_np}
            sio.savemat(os.path.join(out_pred_root, 'class', class_mat_name), class_dict)
            
            #cv2.imwrite(os.path.join(out_pred_root, 'bbox', img_name), predictions)

            if mask_list_binary.shape[-1] > 0:
                masks_no_overlap, bi_map, num_mask = removeoverlap(mask_list_binary)
                out_name = os.path.join(out_pred_root, 'ins_mask', 'binary', img_name.split('.')[0] + '.tif')
                pred_ins = mask2out(masks_no_overlap, num_mask)
                #cv2.imwrite(os.path.join(out_pred_root, 'bi_mask', img_name), (bi_map * 255).astype(np.uint8))
                tiff.imsave(out_name, pred_ins)

            for cat in range(1, 6):
                if mask_list_allcat[cat - 1].shape[-1] > 0:
                    masks_no_overlap, bi_map, num_mask = removeoverlap(mask_list_allcat[cat - 1])
                    out_name = os.path.join(out_pred_root, 'ins_mask', str(cat), img_name.split('.')[0] + '_' + str(cat) + '.tif')
                    pred_ins = mask2out(masks_no_overlap, num_mask)
                    #cv2.imwrite(os.path.join(out_pred_root, 'bi_mask', img_name), (bi_map * 255).astype(np.uint8))
                    tiff.imsave(out_name, pred_ins)
            


if __name__ == "__main__":
    organ_pair = {'breast': 'breast2colon', 'colon': 'colon2breast'}
    source_organ = 'breast'
    target_organ = 'testis'
    gt_rp = False
    
    test_root_name = '/home/jianan/InstanceSeg+Class/raw_datasets/PanNuke/raw/test/' + target_organ + '_images/'
    
    if not gt_rp:
        if target_organ == 'thyroid':
    	    test_root_json = '/home/jianan/InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_thyroid/fold3/breast_train/0064800/instances_mrcn_predict_ins_mask.json'
        elif target_organ == 'bile':
    	    test_root_json = '/home/jianan/InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_bile/fold3/breast_train/0050400/instances_mrcn_predict_ins_mask.json'
        elif target_organ == 'testis':
    	    test_root_json = '/home/jianan/InstanceSeg/main_code/maskrcnn_benchmark/maskrcnn-benchmark/InferenceResult/PanNuke_testis/fold3/breast_train/0057600/instances_mrcn_predict_ins_mask.json'
        else:
    	    raise ValueError("Undefined target organ!!!")
    else:
        test_root_json = '/home/jianan/InstanceSeg+Class/raw_datasets/PanNuke/raw/test/' + target_organ + '_instances_PanNuke_train.json'
    	
    for i in range(1, 11):
        if i < 10:
            model_index = '00%d' % i
        else:
            model_index = '0%d' % i 
        wts_root = '../../checkpoints/mrcn_predict_inpaint_nuc_cor_class/' + source_organ + '2' + target_organ + '_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W/model_epoch_' + model_index + '.pth'
        if gt_rp:
            out_test_root = '../../InferenceResult/mrcn_predict_inpaint_nuc_cor_class/PanNuke_' + target_organ + '/fold3/' + source_organ + '_noRPN_gt_rp' + '_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W/' + model_index + '/'
        else:
            out_test_root = '../../InferenceResult/mrcn_predict_inpaint_nuc_cor_class/PanNuke_' + target_organ + '/fold3/' + source_organ + '_noRPN' + '_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W/' + model_index + '/'
        infer_fluo2tcga(wts_root, test_root_name, test_root_json, out_test_root)
