# Note: The abbreviation ''fmg'' stands for Fine-grained Mask Generator, which is the same as the mask engine in previous AGG.
# GraCo w/ MMT+FMG ViT-B
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py models/plainvit_base448_graco_mmt_fmg.py --part_path /path/to/fmg_proposal.pkl --enable_lora --weights weights/simpleclick/sbd_vit_base.pth --gpus 0,1,2,3

# GraCo w/ MMT+FMG ViT-L
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py models/plainvit_large448_graco_mmt_fmg.py --part_path /path/to/fmg_proposal.pkl --enable_lora --weights weights/simpleclick/sbd_vit_large.pth --gpus 0,1,2,3,4,5,6,7


du -sh * .[^.]*

#修改isegm/engine/trainer.py
## self.freeze_list = cfg.freeze_list if cfg.freeze_list is not None else []
#            self.freeze_list = getattr(cfg, 'freeze_list', [])

#
#GraCo训练命令有效
python train.py models/plainvit_base448_graco.py --weights weights/simpleclick/sbd_vit_base.pth --batch-size=16
python train.py models/plainvit_large448_graco.py --weights weights/simpleclick/sbd_vit_large.pth --batch-size=8
CUDA_VISIBLE_DEVICES=0,1 python train.py models/plainvit_large448_graco.py --weights weights/simpleclick/sbd_vit_large.pth --batch-size=16 --gpus 0,1

#我的任务是提高交互式图像分割精度，主干网络采用了plain vit处理得到了四个不同尺度的特征图[16, 128, 112, 112]、[16, 256, 56, 56]、[16, 512, 28, 28]、[16, 1024, 14, 14]，我想进一步处理特征图以增强信息提取能力，请你详细分析以下代码能否起到帮助作用，请给出你的意见并详细说明


python train.py models/plainvit_base448_graco_mmt_fmg.py --part_path /path/to/fmg_proposal.pkl --enable_lora --weights weights/simpleclick/sbd_vit_base.pth
python train.py models/plainvit_base448_graco_mmt_fmg.py --weights weights/simpleclick/sbd_vit_base.pth --batch-size=16

python demo.py --checkpoint weights\simpleclick\sbd_vit_base.pth --lora_checkpoint weights\GraCo\GraCo_base_lora.pth --gpu 0


python demo.py --checkpoint outputs\plainvit_base448_graco\000\070.pth --lora_checkpoint weights\GraCo\GraCo_base_lora.pth --gpu 0
