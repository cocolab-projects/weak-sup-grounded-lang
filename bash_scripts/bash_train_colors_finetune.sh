cd ~/weak-sup-grounded-lang/scripts

OUT_DIR=$1;
DROPOUT=$2;
SUP_LEV=1.0;
# 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 


##### FINETUNE ######
for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
		screen -S train_colors_${SUP_LEV}_vae_finetune_4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=7 python train_colors_finetune.py ${OUT_DIR}_weaksup_4terms_far ${OUT_DIR}_weaksup_4terms_finetune ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup finetune --context_condition far --cuda; exec bash";
done