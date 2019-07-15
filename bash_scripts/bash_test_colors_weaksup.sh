cd ~/weak-sup-grounded-lang/scripts

LOAD_DIR=$1;
RES_OUT_DIR=$2;

for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
    screen -S test_colors_${SUP_LEV}_weaksup -dm bash -c "CUDA_VISIBLE_DEVICES=6 python test_colors_weaksup.py ${LOAD_DIR} ${RES_OUT_DIR}_test_hard ${SUP_LEV} --num_iter 3 --cuda --hard; exec bash";
    screen -S test_colors_${SUP_LEV}_weaksup_hard -dm bash -c "CUDA_VISIBLE_DEVICES=7 python test_colors_weaksup.py ${LOAD_DIR}_hard ${RES_OUT_DIR}_hard_test_easy ${SUP_LEV} --num_iter 3 --cuda; exec bash";
done
	