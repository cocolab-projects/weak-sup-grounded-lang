cd ~/weak-sup-grounded-lang/scripts

OUT_DIR=$1;
DROPOUT=$2;
SUP_LEV=1.0;

# for ALPHA in 0 0.001 0.01 1 500 1000 2000 5000
# do
# 	for BETA in 1
# 	do
# 		screen -S train_chairs_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_chairs_vae.py ${OUT_DIR}_alpha_beta ${SUP_LEV} --weaksup default --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --epoch 120 --num_iter 3 --context_condition far --cuda; exec bash";
# 	done
# done

# for ALPHA in 1
# do
# 	for BETA in 0 0.0000001 0.0001
# 	do
# 		screen -S train_chairs_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train_chairs_vae.py ${OUT_DIR}_alpha_beta ${SUP_LEV} --weaksup default --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --epoch 120 --num_iter 3 --context_condition far --cuda; exec bash";
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
for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02
do
		screen -S train_chairs_${SUP_LEV}_vae_default -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_chairs_vae.py ${OUT_DIR}_default ${SUP_LEV} --dropout ${DROPOUT} --weaksup default --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
done

for SUP_LEV in 0.05 0.1 0.2 0.5 1.0
do
		screen -S train_chairs_${SUP_LEV}_vae_default -dm bash -c "CUDA_VISIBLE_DEVICES=7 python train_chairs_vae.py ${OUT_DIR}_default ${SUP_LEV} --dropout ${DROPOUT} --weaksup default --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
done

###### 4 terms ######
for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02
do
		screen -S train_chairs_${SUP_LEV}_vae_4terms -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_chairs_vae.py ${OUT_DIR}_weaksup_4terms ${SUP_LEV} --dropout ${DROPOUT} --weaksup 4terms --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
done

for SUP_LEV in 0.05 0.1 0.2 0.5 1.0
do
		screen -S train_chairs_${SUP_LEV}_vae_4terms -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train_chairs_vae.py ${OUT_DIR}_weaksup_4terms ${SUP_LEV} --dropout ${DROPOUT} --weaksup 4terms --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
done

###### 6 terms ######
for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02
do
		screen -S train_chairs_${SUP_LEV}_vae_6terms -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_chairs_vae.py ${OUT_DIR}_weaksup_4terms ${SUP_LEV} --dropout ${DROPOUT} --weaksup 6terms --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
done

for SUP_LEV in 0.05 0.1 0.2 0.5 1.0
do
		screen -S train_chairs_${SUP_LEV}_vae_6terms -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train_chairs_vae.py ${OUT_DIR}_weaksup_4terms ${SUP_LEV} --dropout ${DROPOUT} --weaksup 6terms --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
done

###### post - no unpaired, 4terms ######
# for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02
# do
# 		screen -S train_chairs_${SUP_LEV}_vae_post_nounp_4terms -dm bash -c "CUDA_VISIBLE_DEVICES=0 python train_chairs_vae.py ${OUT_DIR}_weaksup_post_nounp_4terms ${SUP_LEV} --dropout ${DROPOUT} --weaksup post-nounp-4terms --load_dir ${OUT_DIR}_pretrain --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
# done

# for SUP_LEV in 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_chairs_${SUP_LEV}_vae_post_nounp_4terms -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_chairs_vae.py ${OUT_DIR}_weaksup_post_nounp_4terms ${SUP_LEV} --dropout ${DROPOUT} --weaksup post-nounp-4terms --load_dir ${OUT_DIR}_pretrain --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
# done

##### post - unpaired, 4terms ######
# for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02
# do
# 		screen -S train_chairs_${SUP_LEV}_vae_post_unp_4terms -dm bash -c "CUDA_VISIBLE_DEVICES=5 python train_chairs_vae.py ${OUT_DIR}_weaksup_post_unp_4terms ${SUP_LEV} --dropout ${DROPOUT} --weaksup post-4terms --load_dir ${OUT_DIR}_pretrain --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
# done

# for SUP_LEV in 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_chairs_${SUP_LEV}_vae_post_unp_4terms -dm bash -c "CUDA_VISIBLE_DEVICES=6 python train_chairs_vae.py ${OUT_DIR}_weaksup_post_unp_4terms ${SUP_LEV} --dropout ${DROPOUT} --weaksup post-4terms --load_dir ${OUT_DIR}_pretrain --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
# done

###### Pre-training script ######
# screen -S pretrain_chairs__vae -dm bash -c "CUDA_VISIBLE_DEVICES=2 python pretrain_chairs_vae.py ${OUT_DIR}_pretrain --dropout ${DROPOUT} --weaksup pretrain --alpha 1 --beta 2 --num_iter 3 --cuda --context_condition far; exec bash";
