cd ~/weak-sup-grounded-lang/scripts

OUT_DIR=$1;
# SEED=$3;

for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
    screen -S train_chairs_${SUP_LEV}_weaksup_TEST -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train_chairs_weaksup.py ${OUT_DIR}_far_test ${SUP_LEV} --context_condition far --epochs 50 --num_iter 3 --cuda; exec bash";
done
