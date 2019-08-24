cd ~/weak-sup-grounded-lang/scripts

LOAD_DIR=$1;
RES_OUT_DIR=$2;

# 0.0005 0.001 0.002 0.005 0.01

for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
    screen -S test_chairs_${SUP_LEV}_sup_FINAL -dm bash -c "CUDA_VISIBLE_DEVICES=8 python test_chairs_weaksup.py ${LOAD_DIR}_far_final3 ${RES_OUT_DIR}_final3 ${SUP_LEV} --context_condition far --cuda --num_iter 3; exec bash";
done