#!/usr/bin/env bash

source activate $your_env

python tools/train_net.py --config-file "configs/class_breast2thyroid_mrcn_predict.yaml" SOLVER.MAX_ITER 50000 SOLVER.EPOCH_PERIOD 5000 OUTPUT_DIR checkpoints/mrcn_predict_inpaint_nuc_cor_class/breast2thyroid_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W

python tools/train_net.py --config-file "configs/class_breast2testis_mrcn_predict.yaml" SOLVER.MAX_ITER 50000 SOLVER.EPOCH_PERIOD 5000 OUTPUT_DIR checkpoints/mrcn_predict_inpaint_nuc_cor_class/breast2testis_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W

python tools/train_net.py --config-file "configs/class_breast2bile_mrcn_predict.yaml" SOLVER.MAX_ITER 50000 SOLVER.EPOCH_PERIOD 5000 OUTPUT_DIR checkpoints/mrcn_predict_inpaint_nuc_cor_class/breast2bile_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W



python tools/train_net.py --config-file "configs/class_breast2thyroid_13_2_mrcn_predict.yaml" SOLVER.MAX_ITER 50000 SOLVER.EPOCH_PERIOD 5000 OUTPUT_DIR checkpoints/mrcn_predict_inpaint_nuc_cor_class/cross_val/breast2thyroid_13_2_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W

python tools/train_net.py --config-file "configs/class_breast2thyroid_23_1_mrcn_predict.yaml" SOLVER.MAX_ITER 50000 SOLVER.EPOCH_PERIOD 5000 OUTPUT_DIR checkpoints/mrcn_predict_inpaint_nuc_cor_class/cross_val/breast2thyroid_23_1_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W

#python tools/train_net.py --config-file "configs/class_breast2testis_13_2_mrcn_predict.yaml" SOLVER.MAX_ITER 50000 SOLVER.EPOCH_PERIOD 5000 OUTPUT_DIR checkpoints/mrcn_predict_inpaint_nuc_cor_class/cross_val/breast2testis_13_2_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W

python tools/train_net.py --config-file "configs/class_breast2testis_23_1_mrcn_predict.yaml" SOLVER.MAX_ITER 50000 SOLVER.EPOCH_PERIOD 5000 OUTPUT_DIR checkpoints/mrcn_predict_inpaint_nuc_cor_class/cross_val/breast2testis_23_1_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W

python tools/train_net.py --config-file "configs/class_breast2bile_13_2_mrcn_predict.yaml" SOLVER.MAX_ITER 50000 SOLVER.EPOCH_PERIOD 5000 OUTPUT_DIR checkpoints/mrcn_predict_inpaint_nuc_cor_class/cross_val/breast2bile_13_2_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W

python tools/train_net.py --config-file "configs/class_breast2bile_23_1_mrcn_predict.yaml" SOLVER.MAX_ITER 50000 SOLVER.EPOCH_PERIOD 5000 OUTPUT_DIR checkpoints/mrcn_predict_inpaint_nuc_cor_class/cross_val/breast2bile_23_1_PAFPN_1_NUCLEI_FILLER_0.01_NUCLEI_CLASS_0.01_EPOCH_5W
