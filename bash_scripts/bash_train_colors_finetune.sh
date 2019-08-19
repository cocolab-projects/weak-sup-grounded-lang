cd ~/weak-sup-grounded-lang/scripts

OUT_DIR=$1;
DROPOUT=$2;
SUP_LEV=1.0;
# 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 

# for ALPHA in 0 1
# do
# 	for BETA in 0 1 10
# 	do
# 		screen -S train_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=4 python train_colors_vae.py ${OUT_DIR}_alpha_beta_more ${SUP_LEV} --dropout ${DROPOUT} --weaksup default --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
# 	done
# done

# for ALPHA in 0.5 1 5 8 10
# do
# 	for BETA in 10 20 30
# 	do
# 		screen -S train_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_colors_vae.py ${OUT_DIR}_alpha_beta_norm ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
# 	done
# done

# for ALPHA in 5 8 10
# do
# 	for BETA in 1 2 4 5 8
# 	do
# 		screen -S train_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train_colors_vae.py ${OUT_DIR}_alpha_beta_norm ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
# 	done
# done

# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=0 python train_colors_vae.py ${OUT_DIR}_weaksup ${SUP_LEV} --dropout ${DROPOUT} --alpha 0.5 --beta 1000 --num_iter 3 --cuda; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_weaksup_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha 0.5 --beta 1000 --num_iter 3 --cuda --hard; exec bash";
# done

#### WEAK SUPERVISION with unpaired datapoints included VERSION ONLY ####
###### default ######
# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae_default_far -dm bash -c "CUDA_VISIBLE_DEVICES=3 python train_colors_vae.py ${OUT_DIR}_default_far_re ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup default --context_condition far --cuda; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_default_close -dm bash -c "CUDA_VISIBLE_DEVICES=4 python train_colors_vae.py ${OUT_DIR}_default_close_re ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup default --cuda --context_condition close; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_default_all -dm bash -c "CUDA_VISIBLE_DEVICES=5 python train_colors_vae.py ${OUT_DIR}_default_all_re ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup default --cuda --context_condition all; exec bash";
# done

# ##### 4 terms ######
# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae_4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=0 python train_colors_vae.py ${OUT_DIR}_weaksup_4terms_far ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup 4terms --context_condition far --cuda; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_4terms_close -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_weaksup_4terms_close ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup 4terms --cuda --context_condition close; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_4terms_all -dm bash -c "CUDA_VISIBLE_DEVICES=2 python train_colors_vae.py ${OUT_DIR}_weaksup_4terms_all ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup 4terms --cuda --context_condition all; exec bash";
# done

# ##### 6 terms ######
# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae_6terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=3 python train_colors_vae.py ${OUT_DIR}_weaksup_6terms_far ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup 6terms --context_condition far --cuda; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_6terms_close -dm bash -c "CUDA_VISIBLE_DEVICES=4 python train_colors_vae.py ${OUT_DIR}_weaksup_6terms_close ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup 6terms --cuda --context_condition close; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_6terms_all -dm bash -c "CUDA_VISIBLE_DEVICES=5 python train_colors_vae.py ${OUT_DIR}_weaksup_6terms_all ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup 6terms --cuda --context_condition all; exec bash";
# done

##### Pretrain + 4 terms ######
for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
		screen -S train_colors_${SUP_LEV}_vae_finetune_4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train_colors_finetune.py ${OUT_DIR}_weaksup_4terms_far ${OUT_DIR}_weaksup_4terms_finetune ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup finetune --context_condition far --cuda; exec bash";
done

##### Pretrain + 6 terms ######
# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae_postunp6terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_weaksup_post_rev_unp_6terms_far ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-6terms --load_dir ${OUT_DIR}_pretrain_rev_far --context_condition far --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_postunp6terms_close -dm bash -c "CUDA_VISIBLE_DEVICES=4 python train_colors_vae.py ${OUT_DIR}_weaksup_post_unp_6terms_close ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-6terms --load_dir ${OUT_DIR}_pretrain_close --cuda --context_condition close; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_postunp6terms_all -dm bash -c "CUDA_VISIBLE_DEVICES=5 python train_colors_vae.py ${OUT_DIR}_weaksup_post_unp_6terms_all ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-6terms --load_dir ${OUT_DIR}_pretrain_all --cuda --context_condition all; exec bash";
# done

##### Pretrain RGB only + 4 terms ######
# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae_post_only_rgb_unp4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=3 python train_colors_vae.py ${OUT_DIR}_weaksup_post_only_rgb_unp_4terms_far ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-only-rgb-4terms --load_dir ${OUT_DIR}_pretrain_far --seed 50 --context_condition far --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_postunp4terms_close -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_weaksup_post_unp_4terms_close ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-4terms --load_dir ${OUT_DIR}_pretrain_close --cuda --context_condition close; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_postunp4terms_all -dm bash -c "CUDA_VISIBLE_DEVICES=2 python train_colors_vae.py ${OUT_DIR}_weaksup_post_unp_4terms_all ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-4terms --load_dir ${OUT_DIR}_pretrain_all --cuda --context_condition all; exec bash";
# done

##### Pretrain text only + 4 terms ######
# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae_post_only_txt_unp4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=4 python train_colors_vae.py ${OUT_DIR}_weaksup_post_only_text_unp_4terms_far ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-only-text-4terms --load_dir ${OUT_DIR}_pretrain_far --seed 50 --context_condition far --cuda; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_post_rev_only_txt_unp4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=5 python train_colors_vae.py ${OUT_DIR}_weaksup_post_rev_only_text_unp_4terms_far ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-only-text-4terms --load_dir ${OUT_DIR}_pretrain_rev_far --seed 50 --context_condition far --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_postunp4terms_close -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_weaksup_post_unp_4terms_close ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-4terms --load_dir ${OUT_DIR}_pretrain_close --cuda --context_condition close; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_postunp4terms_all -dm bash -c "CUDA_VISIBLE_DEVICES=2 python train_colors_vae.py ${OUT_DIR}_weaksup_post_unp_4terms_all ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-4terms --load_dir ${OUT_DIR}_pretrain_all --cuda --context_condition all; exec bash";
# done

#####***** PRETRAINING BASH *****######
# screen -S train_colors_${SUP_LEV}_vae_pretrain_rev_far -dm bash -c "CUDA_VISIBLE_DEVICES=3 python pretrain_colors_vae.py ${OUT_DIR}_pretrain_rev_far --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup pretrain-reverse --context_condition far --cuda; exec bash";
# screen -S train_colors_${SUP_LEV}_vae_pretrain_rev_close -dm bash -c "CUDA_VISIBLE_DEVICES=4 python pretrain_colors_vae.py ${OUT_DIR}_pretrain_rev_close --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup pretrain-reverse --context_condition close --cuda; exec bash";
# screen -S train_colors_${SUP_LEV}_vae_pretrain_rev_all -dm bash -c "CUDA_VISIBLE_DEVICES=5 python pretrain_colors_vae.py ${OUT_DIR}_pretrain_rev_all --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup pretrain-reverse --context_condition all --cuda; exec bash";

###### Post-train 4terms ######
# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae_post_rev_unp_2terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=2 python train_colors_vae.py ${OUT_DIR}_weaksup_post_rev_nounp_2terms_far2 ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-nounp-2terms --load_dir ${OUT_DIR}_pretrain_rev_far --context_condition far --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_posttrain4terms_close -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_weaksup_posttrain_close2 ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup posttrain --load_dir ${OUT_DIR}_pretrain_close --context_condition close --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_posttrain4terms_all -dm bash -c "CUDA_VISIBLE_DEVICES=2 python train_colors_vae.py ${OUT_DIR}_weaksup_posttrain_all2 ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup posttrain --load_dir ${OUT_DIR}_pretrain_all --context_condition all --cuda; exec bash";
# done

###### Post-train 6terms ######
# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae_post_rev_nounp_4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=2 python train_colors_vae.py ${OUT_DIR}_weaksup_post_rev_nounp_4terms_far2 ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup post-nounp-4terms --load_dir ${OUT_DIR}_pretrain_rev_far --context_condition far --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_posttrain6terms_close -dm bash -c "CUDA_VISIBLE_DEVICES=7 python train_colors_vae.py ${OUT_DIR}_weaksup_posttrain_6terms_close2 ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup posttrain --load_dir ${OUT_DIR}_pretrain_close --context_condition close --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_posttrain6terms_all -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_colors_vae.py ${OUT_DIR}_weaksup_posttrain_6terms_all2 ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup posttrain --load_dir ${OUT_DIR}_pretrain_all --context_condition all --cuda; exec bash";
# done

# ###### Coin-toss by batch ######
# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
# 		screen -S train_colors_${SUP_LEV}_vae_coin_batch_far -dm bash -c "CUDA_VISIBLE_DEVICES=6 python train_colors_vae.py ${OUT_DIR}_weaksup_cointoss_far ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup coin --context_condition far --cuda; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_coin_batch_close -dm bash -c "CUDA_VISIBLE_DEVICES=7 python train_colors_vae.py ${OUT_DIR}_weaksup_cointoss_close ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup coin --context_condition close --cuda; exec bash";
# 		screen -S train_colors_${SUP_LEV}_vae_coin_batch_all -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train_colors_vae.py ${OUT_DIR}_weaksup_cointoss_all ${SUP_LEV} --dropout ${DROPOUT} --alpha 1 --beta 10 --num_iter 3 --weaksup coin --context_condition all --cuda; exec bash";
# done

