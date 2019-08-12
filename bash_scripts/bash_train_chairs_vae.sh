cd ~/weak-sup-grounded-lang/scripts

OUT_DIR=$1;
DROPOUT=$2;
SUP_LEV=1.0;

# for ALPHA in 1
# do
# 	for BETA in 1 2 4 5 8
# 	do
# 		screen -S train_chairs_alpha_${ALPHA}_beta_${BETA}_vae_FIX -dm bash -c "CUDA_VISIBLE_DEVICES=5 python train_chairs_vae.py ${OUT_DIR}_alpha_beta_fix ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --epoch 120 --num_iter 3 --context_condition far --cuda; exec bash";
# 	done
# done

# for ALPHA in 5
# do
# 	for BETA in 1 2 4 5 8
# 	do
# 		screen -S train_chairs_alpha_${ALPHA}_beta_${BETA}_vae_FIX -dm bash -c "CUDA_VISIBLE_DEVICES=7 python train_chairs_vae.py ${OUT_DIR}_alpha_beta_fix ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --epoch 120 --num_iter 3 --context_condition far --cuda; exec bash";
# 	done
# done

# for ALPHA in 10
# do
# 	for BETA in 1 2 4 5 8
# 	do
# 		screen -S train_chairs_alpha_${ALPHA}_beta_${BETA}_vae_FIX -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_chairs_vae.py ${OUT_DIR}_alpha_beta_fix ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --epoch 120 --num_iter 3 --context_condition far --cuda; exec bash";
# 	done
# done

#### WEAK SUPERVISION with unpaired datapoints included VERSION ONLY ####
###### default ######
# for SUP_LEV in 0.0005 0.001 0.002 0.005
# do
# 		screen -S train_chairs_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_chairs_vae.py ${OUT_DIR}_default ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
# done

# for SUP_LEV in 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_chairs_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=7 python train_chairs_vae.py ${OUT_DIR}_default ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
# done

###### 4 terms ######
for SUP_LEV in 0.0005 0.001 0.002 0.005
do
		screen -S train_chairs_${SUP_LEV}_vae_4terms -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_chairs_vae.py ${OUT_DIR}_default ${SUP_LEV} --dropout ${DROPOUT} --weaksup 4terms --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
done

for SUP_LEV in 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
		screen -S train_chairs_${SUP_LEV}_vae_4terms -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train_chairs_vae.py ${OUT_DIR}_default ${SUP_LEV} --dropout ${DROPOUT} --weaksup 4terms --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
done


