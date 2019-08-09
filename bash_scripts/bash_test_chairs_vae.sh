cd ~/weak-sup-grounded-lang/scripts

LOAD_DIR=$1;
RES_OUT_DIR=$2;
SUP_LEV=1.0;

for ALPHA in 1
do
	for BETA in 1 2 4 5 8
	do
		screen -S test_chairs_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=5 python test_chairs_vae.py ${LOAD_DIR}_alpha_beta_epoch ${RES_OUT_DIR}_alpha_beta_epoch --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --context_condition far --cuda; exec bash";
		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
	done
done

for ALPHA in 5
do
	for BETA in 1 2 4 5 8
	do
		screen -S test_chairs_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=7 python test_chairs_vae.py ${LOAD_DIR}_alpha_beta_epoch ${RES_OUT_DIR}_alpha_beta_epoch --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --context_condition far --cuda; exec bash";
		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
	done
done

for ALPHA in 10
do
	for BETA in 1 2 4 5 8
	do
		screen -S test_chairs_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=8 python test_chairs_vae.py ${LOAD_DIR}_alpha_beta_epoch ${RES_OUT_DIR}_alpha_beta_epoch --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --context_condition far --cuda; exec bash";
		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
	done
done

# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
#     screen -S test_colors_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=6 python test_colors_vae_exp.py ${LOAD_DIR}_weaksup ${RES_OUT_DIR}_weaksup_exp --sup_lvl ${SUP_LEV} --alpha 1 --beta 10 --num_iter 1 --cuda; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae_test_hard -dm bash -c "CUDA_VISIBLE_DEVICES=7 python test_colors_vae_exp.py ${LOAD_DIR}_weaksup ${RES_OUT_DIR}_weaksup_test_hard_exp --sup_lvl ${SUP_LEV} --alpha 1 --beta 10 --num_iter 1 --cuda --context_condition all; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=8 python test_colors_vae_exp.py ${LOAD_DIR}_weaksup_hard ${RES_OUT_DIR}_weaksup_hard_exp --sup_lvl ${SUP_LEV} --alpha 1 --beta 10 --num_iter 1 --cuda; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae_hard_test_hard -dm bash -c "CUDA_VISIBLE_DEVICES=9 python test_colors_vae_exp.py ${LOAD_DIR}_weaksup_hard ${RES_OUT_DIR}_weaksup_hard_test_hard_exp --sup_lvl ${SUP_LEV} --alpha 1 --beta 10 --num_iter 1 --cuda --context_condition all; exec bash";
# done

# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
#     screen -S test_colors_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=0 python test_colors_vae.py ${LOAD_DIR}/ ${RES_OUT_DIR} ${SUP_LEV} --num_iter 3 --cuda; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python test_colors_vae.py ${LOAD_DIR}_hard/ ${RES_OUT_DIR}_hard ${SUP_LEV} --num_iter 3 --cuda; exec bash";
# done
