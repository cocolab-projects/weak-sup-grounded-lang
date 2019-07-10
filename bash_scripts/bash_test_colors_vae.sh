cd ~/weak-sup-grounded-lang/scripts

LOAD_DIR=$1;
RES_OUT_DIR=$2;
SUP_LEV=1.0;

for ALPHA in 0.1 0.2 0.5
do
	for BETA in 35 40 50 60 80
	do
		screen -S test_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=0 python test_colors_vae.py ${LOAD_DIR}/ ${RES_OUT_DIR} --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
	done
done

for ALPHA in 0.05 0.3 0.4
do
	for BETA in 40 50 80 100 120
	do
		screen -S test_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=1 python test_colors_vae.py ${LOAD_DIR}/ ${RES_OUT_DIR} --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
	done
done

for ALPHA in 0.1 0.5
do
	for BETA in 40 100 200 400 500 1000
	do
		screen -S test_colors_alpha_${ALPHA}_beta_${BETA}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=2 python test_colors_vae.py ${LOAD_DIR}/ ${RES_OUT_DIR} --sup_lvl ${SUP_LEV} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda; exec bash";
		# screen -S train_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train_colors_vae.py ${OUT_DIR}_hard ${SUP_LEV} --dropout ${DROPOUT} --alpha ${ALPHA} --beta ${BETA} --num_iter 3 --cuda --hard; exec bash";
	done
done

# for SUP_LEV in 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
# do
#     screen -S test_colors_${SUP_LEV}_vae -dm bash -c "CUDA_VISIBLE_DEVICES=0 python test_colors_vae.py ${LOAD_DIR}/ ${RES_OUT_DIR} ${SUP_LEV} --num_iter 3 --cuda; exec bash";
#     screen -S test_colors_${SUP_LEV}_vae_hard -dm bash -c "CUDA_VISIBLE_DEVICES=1 python test_colors_vae.py ${LOAD_DIR}_hard/ ${RES_OUT_DIR}_hard ${SUP_LEV} --num_iter 3 --cuda; exec bash";
# done
