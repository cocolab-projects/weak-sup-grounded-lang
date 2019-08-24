cd ~/weak-sup-grounded-lang/scripts

LOAD_DIR=$1;
RES_OUT_DIR=$2;
SUP_LEV=1.0;


for SUP_LEV in 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
		screen -S test_chairs_${SUP_LEV}_vae_finetune_4terms_far -dm bash -c "CUDA_VISIBLE_DEVICES=7 python test_chairs_finetune.py ${LOAD_DIR}_weaksup_4terms_finetune ${RES_OUT_DIR}_weaksup_4terms_finetune --sup_lvl ${SUP_LEV} --alpha 1 --beta 2 --num_iter 3 --context_condition far --cuda; exec bash";
done