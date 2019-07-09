cd ~/weak-sup-grounded-lang/scripts

OUT_DIR=$1;
# SEED=$2;

for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
    screen -S train_colors_${SUP_LEV}_weaksup_paired -dm bash -c "CUDA_VISIBLE_DEVICES=0 python train_colors_weaksup_paired.py ${OUT_DIR} ${SUP_LEV} --num_iter 3 --cuda; exec bash";
    screen -S train_colors_${SUP_LEV}_weaksup_paired_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_weaksup_paired.py ${OUT_DIR}_hard ${SUP_LEV} --num_iter 3 --cuda --hard; exec bash";
done
