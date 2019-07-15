cd ~/weak-sup-grounded-lang/scripts

LOAD_DIR=$1;
RES_OUT_DIR=$2;

for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
    screen -S test_colors_${SUP_LEV}_weaksup_paired -dm bash -c "CUDA_VISIBLE_DEVICES=8 python test_colors_weaksup_paired.py ${LOAD_DIR} ${RES_OUT_DIR}_test_hard ${SUP_LEV} --cuda --num_iter 3 --hard; exec bash";
    screen -S test_colors_${SUP_LEV}_weaksup_paired_hard -dm bash -c "CUDA_VISIBLE_DEVICES=9 python test_colors_weaksup_paired.py ${LOAD_DIR}_hard ${RES_OUT_DIR}_hard_test_easy ${SUP_LEV} --cuda --num_iter 3; exec bash";
done
