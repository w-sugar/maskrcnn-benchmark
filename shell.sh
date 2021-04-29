CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 1111 tools/train_net.py --config-file configs/e2e_cascade_rcnn_R2plus1d_34_FPN_1x.yaml
