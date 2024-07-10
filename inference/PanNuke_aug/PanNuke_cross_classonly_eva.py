
import os
import glob
import numpy as np
import tifffile as tiff
import scipy.io as sio
from metrics import agg_jc_index, pixel_f1, remap_label_sizethresh, get_fast_pq, pair_coordinates
import xlwt


def evaluate_fluo2tcga(pred_root, gt_root, txt_fp, min_size):
    #aji_list = []
    #f1_list = []
    pq_list = []

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Test Sheet')

    counter = 0

    ws.write(0, 0, 'img_name')
    #ws.write(0, 1, 'aji')
    #ws.write(0, 2, 'f1')
    ws.write(0, 1, 'pq')

    # for img_name in img_names:
    image_list = sorted(os.listdir(gt_root))
    image_list = [os.path.splitext(image)[0] for image in image_list]
    pred_list = sorted(os.listdir(pred_root))
    pred_list = [os.path.splitext(image)[0] for image in pred_list]
    union_list = list(set(image_list).union(set(pred_list)))
    intersection_list = list(set(image_list).intersection(set(pred_list)))
    unmatched_num = len(union_list) - len(intersection_list)

    for image in image_list:
        gt_name = image + '.tif'
        img_name = image + '.tif'
        if os.path.exists(os.path.join(pred_root, img_name)) == False:
            continue
        pred_ins = tiff.imread(os.path.join(pred_root, img_name))
        gt_ins = tiff.imread(os.path.join(gt_root, gt_name))
        
        if pred_ins.shape != gt_ins.shape:
            continue

        # compute predictions
        gt = remap_label_sizethresh(gt_ins, 0)
        pred = remap_label_sizethresh(pred_ins, min_size)

        # object level pq
        pq_info_cur = get_fast_pq(gt, pred, match_iou=0.5)[0]
        pq_cur = pq_info_cur[2]
        pq_list.append(pq_cur)

        # object-level aji
        #aji_cur = agg_jc_index(gt, pred)
        #aji_list.append(aji_cur)

        # pixel level dice/f1 score
        #f1_cur = pixel_f1(gt, pred)
        #f1_list.append(f1_cur)

        counter = counter + 1

        ws.write(counter, 0, img_name)
        #ws.write(counter, 1, str(aji_cur))
        #ws.write(counter, 2, str(f1_cur))
        ws.write(counter, 1, str(pq_cur))

        #print('The evaluation for current image:', img_name, 'is: aji score: ', aji_cur, 'f1 score: ', f1_cur)
        #txt_fp.writelines(
        #    'The evaluation for current image: ' + img_name + ' is: aji score: ' + str(aji_cur) + 'f1 score: ' + str(
        #        f1_cur) + '\n')

    for num in range(unmatched_num):
        pq_list.append(0.0)
    #aji_array = np.asarray(aji_list, dtype=np.float16)
    #f1_array = np.asarray(f1_list, dtype=np.float16)
    pq_array = np.asarray(pq_list, dtype=np.float16)

    #aji_avg = np.average(aji_array)
    #aji_std = np.std(aji_array)

    #f1_avg = np.average(f1_array)
    #f1_std = np.std(f1_array)

    pq_avg = np.average(pq_array)
    pq_std = np.std(pq_array)

    ws.write(counter + 1, 0, 'Average')
    #ws.write(counter + 1, 1, str(aji_avg))
    #ws.write(counter + 1, 2, str(f1_avg))
    ws.write(counter + 1, 1, str(pq_avg))

    ws.write(counter + 2, 0, 'Standard deviation')
    #ws.write(counter + 2, 1, str(aji_std))
    #ws.write(counter + 2, 2, str(f1_std))
    ws.write(counter + 2, 1, str(pq_std))

    wb.save(pred_root + 'eval_dngt_minsize_' + str(min_size) + '.xls')

    print(pred_root)
    txt_fp.writelines(pred_root + '\n')

    #print('average aji score of this method is: ', aji_avg, ' ', aji_std)
    #print('average f1 score of this method is: ', f1_avg, ' ', f1_std)
    print('average pq score of this method is: ', pq_avg, ' ', pq_std)
    #txt_fp.writelines('average aji score of this method is: ' + str(aji_avg) + ' ' + str(aji_std) + '\n')
    #txt_fp.writelines('average f1 score of this method is: ' + str(f1_avg) + ' ' + str(f1_std) + '\n')
    txt_fp.writelines('average pq score of this method is: ' + str(pq_avg) + ' ' + str(pq_std) + '\n')


def run_nuclei_type_stat(pred_dir, true_dir, txt_fp, type_uid_list=[1, 2, 3, 5], exhaustive=True):
    """GT must be exhaustively annotated for instance location (detection).
    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image.
                            Each .mat must contain:
                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index
                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance
        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.
        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types

    """
    file_list = glob.glob(pred_dir + "*.mat")
    file_list.sort()  # ensure same order [1]

    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point
    for file_idx, filename in enumerate(file_list[:]):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        true_info = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        # dont squeeze, may be 1 instance exist
        true_centroid = (true_info["inst_centroid"]).astype("float32")
        true_inst_type = (true_info["inst_type"]).astype("int32")

        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:, 0]
        else:  # no instance at all
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        # * for converting the GT type in CoNSeP
        # true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 3
        # true_inst_type[(true_inst_type == 5) | (true_inst_type == 6) | (true_inst_type == 7)] = 4

        pred_info = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        # dont squeeze, may be 1 instance exist
        pred_centroid = (pred_info["inst_centroid"]).astype("float32")
        pred_inst_type = (pred_info["inst_type"]).astype("int32")

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:, 0]
        else:  # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, 12
        )

        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
                2 * (tp_dt + tn_dt)
                + w[0] * fp_dt
                + w[1] * fn_dt
                + w[2] * fp_d
                + w[3] * fn_d
        )
        return f1_type
        #return [tp_dt, tn_dt, fp_dt, fn_dt, fp_d, fn_d]

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    w = [2, 2, 0, 0]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

    results_list = [f1_d, acc_type]
    for type_uid in type_uid_list:
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

    txt_fp.writelines(pred_dir + '\n')
    txt_fp.writelines('Detection Quality: ' + str(results_list[0]) + '\n')
    
    txt_fp.writelines('Neoplastic Classification: ' + str(results_list[2]) + '\n')
    txt_fp.writelines('Inflammatory Classification: ' + str(results_list[3]) + '\n')
    txt_fp.writelines('Connective Classification: ' + str(results_list[4]) + '\n')
    txt_fp.writelines('Epithelial Classification: ' + str(results_list[5]) + '\n')
    txt_fp.writelines('Classification Avg: ' + str((results_list[2]+results_list[3]+results_list[4]+results_list[5])/4) + '\n')



if __name__ == "__main__":
    min_size = 50
    source_organ = 'breast'
    target_organ = 'testis'
    tail_name = '_noRPN_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W'
    #tail_name = '_noRPN_gt_rp_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W'
    
    for cat in ['1', '2', '3', '5', 'binary']:
        test_gt_root = '/home/jianan/InstanceSeg+Class/raw_datasets/PanNuke/raw/test/' + target_organ + '_mask_patches/' + cat + '/'
        txt_path = '../../InferenceResult/mrcn_predict_inpaint_nuc_cor_class/PanNuke_' + target_organ + '/fold3/' + source_organ + tail_name + '/eva_strict_' + cat + '.txt'
        fp = open(txt_path, "w")
        for i in range(1, 11):
            if i < 10:
                model_index = '00%d' % i
            else:
                model_index = '0%d' % i
            test_pred_root = '../../InferenceResult/mrcn_predict_inpaint_nuc_cor_class/PanNuke_' + target_organ + '/fold3/' + source_organ + tail_name + '/' + model_index + '/ins_mask/' + cat + '/'
            evaluate_fluo2tcga(test_pred_root, test_gt_root, fp, min_size)
        fp.close()
    
    
    fp = open('../../InferenceResult/mrcn_predict_inpaint_nuc_cor_class/PanNuke_' + target_organ + '/fold3/' + source_organ + tail_name + '/eva_strict_avg.txt', 'w')
    cat_pq = np.zeros([4, 10])
    for idx, cat in enumerate(['1', '2', '3', '5']):
        cat_txt = open('../../InferenceResult/mrcn_predict_inpaint_nuc_cor_class/PanNuke_' + target_organ + '/fold3/' + source_organ + tail_name + '/eva_strict_' + cat + '.txt', 'r')
        cat_lines = cat_txt.readlines()
        for model in range(10):
            pq_value = float(cat_lines[model * 2 + 1].split(': ')[-1].split(' ')[0])
            cat_pq[idx, model] = pq_value
        cat_txt.close()
    cat_pq = np.mean(cat_pq, axis=0)
    for model in range(10):
        fp.writelines(cat_lines[model * 2])
        fp.writelines('class-wise average pq score of this method is: ' + str(cat_pq[model]) + '\n')
    fp.close()
    
    
    fp = open('../../InferenceResult/mrcn_predict_inpaint_nuc_cor_class/PanNuke_' + target_organ + '/fold3/' + source_organ + tail_name + '/eva_class_nodet.txt', "w")
    true_dir = '/home/jianan/InstanceSeg+Class/raw_datasets/PanNuke/raw/test/' + target_organ + '_mask_patches/class/'
    for i in range(1, 11):
        if i < 10:
            model_index = '00%d' % i
        else:
            model_index = '0%d' % i
        pred_dir = '../../InferenceResult/mrcn_predict_inpaint_nuc_cor_class/PanNuke_' + target_organ + '/fold3/' + source_organ + tail_name + '/' + model_index + '/class/'
        run_nuclei_type_stat(pred_dir, true_dir, fp)
    fp.close()
    

    
