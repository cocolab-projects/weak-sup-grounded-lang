cd ~/weak-sup-grounded-lang/scripts

LOAD_DIR=$1;
RES_OUT_DIR=$2;
SUP_LEV=1.0;

for ALPHA in 0.5 1 5
do
	for BETA in 1 2 4 5 8
	do
		screen -S test_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=1 python test_colors_vae.py ${LOAD_DIR}_alpha_beta ${RES_OUT_DIR}_alpha_beta_test --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
	done
done

for ALPHA in 0.5 1 5 8 10
do
	for BETA in 10 20 30
	do
		screen -S test_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=2 python test_colors_vae.py ${LOAD_DIR}_alpha_beta ${RES_OUT_DIR}_alpha_beta_test --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
	done
done

for ALPHA in 8 10
do
	for BETA in 1 2 4 5 8
	do
		screen -S test_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=3 python test_colors_vae.py ${LOAD_DIR}_alpha_beta ${RES_OUT_DIR}_alpha_beta_test --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
	done
done

for ALPHA in 2
do
	for BETA in 1 2 4 5 8 10 20 30
	do
		screen -S test_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=4 python test_colors_vae.py ${LOAD_DIR}_alpha_beta ${RES_OUT_DIR}_alpha_beta_test --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
	done
done

# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
#     screen -S test_colors_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=3 python test_colors_vae.py ${LOAD_DIR}_weaksup_2 ${RES_OUT_DIR}_weaksup_2 --sup_lvl ${SUP_LEV} --alpha 2 --beta 10 --num_iter 3 --cuda; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=4 python test_colors_vae.py ${LOAD_DIR}_weaksup_2 ${RES_OUT_DIR}_weaksup_2_test_hard --sup_lvl ${SUP_LEV} --alpha 2 --beta 10 --num_iter 3 --cuda --hard; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=5 python test_colors_vae.py ${LOAD_DIR}_weaksup_hard_2/ ${RES_OUT_DIR}_weaksup_hard_2 --sup_lvl ${SUP_LEV} --alpha 2 --beta 10 --num_iter 3 --cuda; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=6 python test_colors_vae.py ${LOAD_DIR}_weaksup_hard_2/ ${RES_OUT_DIR}_weaksup_hard_2_test_hard --sup_lvl ${SUP_LEV} --alpha 2 --beta 10 --num_iter 3 --cuda --hard; exec bash";
# done

# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
#     screen -S test_colors_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=0 python test_colors_vae.py ${LOAD_DIR}/ ${RES_OUT_DIR} ${SUP_LEV} --num_iter 3 --cuda; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python test_colors_vae.py ${LOAD_DIR}_hard/ ${RES_OUT_DIR}_hard ${SUP_LEV} --num_iter 3 --cuda; exec bash";
# done
