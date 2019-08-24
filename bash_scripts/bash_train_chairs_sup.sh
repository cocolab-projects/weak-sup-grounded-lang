cd ~/weak-sup-grounded-lang/scripts

OUT_DIR=$1;
# SEED=$3;

for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
    screen -S train_chairs_${SUP_LEV}_weaksup_FINAL -dm bash -c "CUDA_VISIBLE_DEVICES=7 python train_chairs_weaksup.py ${OUT_DIR}_far_final3 ${SUP_LEV} --context_condition all --seed 42 --epochs 100 --num_iter 3 --cuda; exec bash";
done
