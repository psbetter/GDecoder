CUDA_VISIBLE_DEVICES=5 python train.py \
--content train_gdecoder_d \
--crop-size 384 \
--data-dir /home/ubuntu/jyt/datasets/FSC-147 \
--batch-size 4 \
--epochs 200 \
--lr 1e-4 \
--weight-decay 0.05 \
--resume /home/ubuntu/jyt/checkpoints/mae/mae_pretrain_vit_base_full.pth \
--scale 60