export segvol_ckpt="path/to/medsam_model_e500.pth"
export work_dir="path/to/work_dir"
export demo_config_path="./config/config_demo.json"

CUDA_VISIBLE_DEVICES=0 python inference_demo.py \
--resume $segvol_ckpt \
-work_dir $work_dir \
--demo_config $demo_config_path 
