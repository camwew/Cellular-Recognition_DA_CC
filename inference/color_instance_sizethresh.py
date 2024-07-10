#from colormap import colormap
import cv2
import numpy as np
import os
import tifffile as tiff
import scipy.io as scio

import numpy as np


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
        #print('not rgb')
    return color_list
    
    
def mkdir(out_folder):
    try:
        os.stat(os.path.dirname(out_folder + '/'))
    except:
        os.makedirs(os.path.dirname(out_folder + '/'))

def color_instance(img_folder, ins_folder, out_folder, min_size):

    mkdir(out_folder)
    imglist = [ins_name for ins_name in os.listdir(ins_folder) if os.path.splitext(ins_name)[1] == '.tif']
    #imglist = [ins_name for ins_name in os.listdir(ins_folder) if os.path.splitext(ins_name)[1] == '.mat']

    for imgname in imglist:

        imgpath = os.path.join(img_folder, imgname.split('.')[0] + '.png')
        inspath = os.path.join(ins_folder, imgname)

        ins_seg = tiff.imread(inspath)
        #ins_seg = scio.loadmat(inspath)['inst_map']
        img = cv2.imread(imgpath)

        masknum = np.amax(ins_seg)
        #print(masknum)

        color_list = colormap()

        for idx in range(1, masknum + 1):
            color_mask = color_list[idx % len(color_list), 0:3]

            # bi_mask_map = (ins_seg == idx).astype(np.uint8)
            bi_mask_map = (ins_seg == idx)
            size = np.sum(bi_mask_map)
            if size <= min_size:
                continue

            ins = np.nonzero(bi_mask_map)

            img = img.astype(np.float64)
            img[ins[0], ins[1], :] *= 0.3
            img[ins[0], ins[1], :] += 0.7 * color_mask

        out_name = imgname.split('.')[0] + '.png'
        cv2.imwrite(os.path.join(out_folder, out_name), img.astype(np.uint8))



