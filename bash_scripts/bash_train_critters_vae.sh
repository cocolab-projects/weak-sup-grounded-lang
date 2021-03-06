cd ~/weak-sup-grounded-lang/scripts

OUT_DIR=$1;
DROPOUT=$2;
SUP_LEV=1.0;

for ALPHA in 1
do
	for BETA in 1 2 4 5 8
	do
		screen -S train_critters_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=3 python train_chairs_vae.py ${OUT_DIR}_alpha_beta ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --dataset critters --epoch 120 --num_iter 3 --context_condition far --cuda; exec bash";
	done
done

for ALPHA in 5
do
	for BETA in 1 2 4 5 8
	do
		screen -S train_critters_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=4 python train_chairs_vae.py ${OUT_DIR}_alpha_beta ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --dataset critters --epoch 120 --num_iter 3 --context_condition far --cuda; exec bash";
	done
done

for ALPHA in 10
do
	for BETA in 1 2 4 5 8
	do
		screen -S train_critters_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=5 python train_chairs_vae.py ${OUT_DIR}_alpha_beta ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --dataset critters --epoch 120 --num_iter 3 --context_condition far --cuda; exec bash";
	done
done

# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=6 python train_colors_vae.py ${OUT_DIR}_weaksup_2_re ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --cuda; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=7 python train_colors_vae.py ${OUT_DIR}_weaksup_hard_2_re ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --cuda --hard; exec bash";
# done


# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=0 python train_colors_vae.py ${OUT_DIR}_weaksup ${SUP_LEV} --dropout ${DROPOUT} --alpha 0.5 --beta 1000 --num_iter 3 --cuda; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_weaksup_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha 0.5 --beta 1000 --num_iter 3 --cuda --hard; exec bash";
# done
