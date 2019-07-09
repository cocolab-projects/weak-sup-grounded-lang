cd scripts

OUT_DIR=$1;
ACC_OUT_DIR=$2;

for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
    screen -S colors_${SUP_LEV}_supervised -dm bash -c "CUDA_VISIBLE_DEVICES=0 python test_colors_vae.py ${OUT_DIR} ${ACC_OUT_DIR} --sup_lvl ${SUP_LEV} --num_iter 3; exec bash";
done