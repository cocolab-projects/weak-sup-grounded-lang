cd ~/weak-sup-grounded-lang/scripts

LOAD_DIR=$1;
RES_OUT_DIR=$2;
SUP_LEV=1.0;

# for ALPHA in 0 1
# do
# 	for BETA in 0 1 10
# 	do
# 		screen -S test_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=4 python test_colors_vae.py ${LOAD_DIR}_alpha_beta_more ${RES_OUT_DIR}_alpha_beta_more --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
# 	done
# done

# for ALPHA in 0.5 1 5 8 10
# do
# 	for BETA in 10 20 30
# 	do
# 		screen -S test_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=8 python test_colors_vae.py ${LOAD_DIR}_weaksup ${RES_OUT_DIR}_weaksup --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 1 --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
# 	done
# done

# for ALPHA in 5 8 10
# do
# 	for BETA in 1 2 4 5 8
# 	do
# 		screen -S test_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=9 python test_colors_vae\.py ${LOAD_DIR}_weaksup ${RES_OUT_DIR}_weaksup --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 1 --cuda; exec bash";
# 		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
# 	done
# done

# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
#     screen -S test_colors_${SUP_LEV}_vae_post_only_text_unp_4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=7 python test_colors_vae.py ${LOAD_DIR}_weaksup_post_only_text_unp_4terms_far ${RES_OUT_DIR}_weaksup_post_only_text_unp_4terms_far-far2 --sup_lvl ${SUP_LEV} --alpha 1 --beta 10 --num_iter 3 --context_condition far --cuda; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae_post_only_rgb_unp_4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=8 python test_colors_vae.py ${LOAD_DIR}_weaksup_post_only_rgb_unp_4terms_far ${RES_OUT_DIR}_weaksup_post_only_rgb_unp_4terms_far-far2 --sup_lvl ${SUP_LEV} --alpha 1 --beta 10 --num_iter 3 --context_condition far --cuda; exec bash";
# 	screen -S test_colors_${SUP_LEV}_vae_post_rev_only_text_unp_4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=9 python test_colors_vae.py ${LOAD_DIR}_weaksup_post_rev_only_text_unp_4terms_far ${RES_OUT_DIR}_weaksup_post_rev_only_text_unp_4terms_far-far --sup_lvl ${SUP_LEV} --alpha 1 --beta 10 --num_iter 3 --context_condition far --cuda; exec bash";
# 	# screen -S test_colors_${SUP_LEV}_vae_post_rev_nounp_2terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=1 python test_colors_vae.py ${LOAD_DIR}_weaksup_post_rev_nounp_2terms_far2 ${RES_OUT_DIR}_weaksup_post_rev_nounp_2terms_far-far2 --sup_lvl ${SUP_LEV} --alpha 1 --beta 10 --num_iter 3 --context_condition far --cuda; exec bash";
# 	# screen -S test_colors_${SUP_LEV}_vae_post_rev_nounp_4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=1 python test_colors_vae.py ${LOAD_DIR}_weaksup_post_rev_nounp_4terms_far2 ${RES_OUT_DIR}_weaksup_post_rev_nounp_4terms_far-far2 --sup_lvl ${SUP_LEV} --alpha 1 --beta 10 --num_iter 3 --context_condition far --cuda; exec bash";
# done

for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
    screen -S test_colors_${SUP_LEV}_vae_4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=7 python test_colors_vae.py ${LOAD_DIR}_weaksup_4terms_far ${RES_OUT_DIR}_weaksup_4terms_re --sup_lvl ${SUP_LEV} --alpha 1 --beta 10 --num_iter 3 --cuda --context_condition far; exec bash";
done

# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
#     screen -S test_colors_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=0 python test_colors_vae.py ${LOAD_DIR}/ ${RES_OUT_DIR} ${SUP_LEV} --num_iter 3 --cuda; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python test_colors_vae.py ${LOAD_DIR}_hard/ ${RES_OUT_DIR}_hard ${SUP_LEV} --num_iter 3 --cuda; exec bash";
# done
