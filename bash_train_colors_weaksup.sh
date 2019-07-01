cd scripts
conda activate pytorch4;

OUT_DIR=$1;
LOAD_DIR=$3;

for SUP_LEV in 0.0005 0.001 0.005 0.01 0.02 0.05 0.1 0.5 1
do
    screen -S colors_${SUP_LEV}_supervised -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_colors_weaksup.py ${OUTDIR} --sup_lvl ${SUP_LEV} --epoch ${EPOCH} --num_iter 3 --cuda;
    python test_colors.py ${LOAD_DIR} ${OUT_DIR} --sup_lvl ${SUP_LEV} --num_iter 3; exec bash";
done