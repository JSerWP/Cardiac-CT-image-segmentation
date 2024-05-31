import argparse
import os
import torch
import torch.nn.functional as F
import json
import monai.transforms as transforms

from model.segment_anything_volumetric import sam_model_registry
from model.network.model import SegVol
from model.data_process.demo_data_process import process_ct_gt
from model.utils.monai_inferers_utils import sliding_window_inference, generate_box, select_points, build_binary_cube, build_binary_points, logits2roi_coor
from model.utils.visualize import draw_result
import streamlit as st

def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", default=True, type=bool)
    parser.add_argument("--resume", type = str, default = 'C:/Users/WangHL/Desktop/Heart_SegVol/SegVol/medsam_model_e500.pth')
    parser.add_argument("-infer_overlap", default=0.0, type=float, help="sliding window inference overlap")
    parser.add_argument("-spatial_size", default=(32, 256, 256), type=tuple)
    parser.add_argument("-patch_size", default=(4, 16, 16), type=tuple)
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    ### demo
    parser.add_argument("--clip_ckpt", type = str, default = 'model/config/clip')
    args = parser.parse_args()
    return args

def zoom_in_zoom_out(args, segvol_model, image, image_resize, text_prompt, point_prompt, box_prompt):
    image_single_resize = image_resize
    image_single = image[0,0]
    ori_shape = image_single.shape
    resize_shape = image_single_resize.shape[2:]
    
    # generate prompts
    text_single = None if text_prompt is None else [text_prompt]
    points_single = None
    box_single = None

    if args.use_point_prompt:
        point, point_label = point_prompt
        points_single = (point.unsqueeze(0).float(), point_label.unsqueeze(0).float()) 
        binary_points_resize = build_binary_points(point, point_label, resize_shape)
    if args.use_box_prompt:
        box_single = box_prompt.unsqueeze(0).float()
        binary_cube_resize = build_binary_cube(box_single, binary_cube_shape=resize_shape)
    
    ####################
    # zoom-out inference:
    print('--- zoom out inference ---')
    print(text_single)
    print(f'use text-prompt [{text_single!=None}], use box-prompt [{box_single!=None}], use point-prompt [{points_single!=None}]')
    with torch.no_grad():
        logits_global_single = segvol_model(image_single_resize,
                                            text=text_single, 
                                            boxes=box_single, 
                                            points=points_single)
    # for key in score.keys():
    #     if key == 'res':
    #         print(f'The mask is {score[key]}')
    #         st.session_state.semantic_res = score[key]
    #     else:
    #         print(f'{key} scored {score[key]:.4f}')
    # resize back global logits
    logits_global_single = F.interpolate(
            logits_global_single.cpu(),
            size=ori_shape, mode='nearest')[0][0]
    
    # build prompt reflection for zoom-in
    if args.use_point_prompt:
        binary_points = F.interpolate(
            binary_points_resize.unsqueeze(0).unsqueeze(0).float(),
            size=ori_shape, mode='nearest')[0][0]
    if args.use_box_prompt:
        binary_cube = F.interpolate(
            binary_cube_resize.unsqueeze(0).unsqueeze(0).float(),
            size=ori_shape, mode='nearest')[0][0]
    # draw_result('unknow', image_single_resize, None, point_prompt, logits_global_single, logits_global_single)
    if not args.use_zoom_in:
        return logits_global_single

    ####################
    # zoom-in inference:
    min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(args.spatial_size, logits_global_single)
    if min_d is None:
        print('Fail to detect foreground!')
        return logits_global_single

    # Crop roi
    image_single_cropped = image_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1].unsqueeze(0).unsqueeze(0)
    global_preds = (torch.sigmoid(logits_global_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1])>0.5).long()
    
    assert not (args.use_box_prompt and args.use_point_prompt)
    # label_single_cropped = label_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1].unsqueeze(0).unsqueeze(0)
    prompt_reflection = None
    if args.use_box_prompt:
        binary_cube_cropped = binary_cube[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
        prompt_reflection = (
            binary_cube_cropped.unsqueeze(0).unsqueeze(0),
            global_preds.unsqueeze(0).unsqueeze(0)
        )
    if args.use_point_prompt:
        binary_points_cropped = binary_points[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
        prompt_reflection = (
            binary_points_cropped.unsqueeze(0).unsqueeze(0),
            global_preds.unsqueeze(0).unsqueeze(0)
        )

    ## inference
    with torch.no_grad():
        logits_single_cropped = sliding_window_inference(
                image_single_cropped, prompt_reflection,
                args.spatial_size, 1, segvol_model, args.infer_overlap,
                text=text_single,
                use_box=args.use_box_prompt,
                use_point=args.use_point_prompt,
                logits_global_single=logits_global_single,
            )
        logits_single_cropped = logits_single_cropped.cpu().squeeze()
        if logits_single_cropped.shape != logits_global_single.shape:
            logits_global_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1] = logits_single_cropped

    return logits_global_single

@st.cache_resource
def build_model():
    # build model
    st.write('building model')
    clip_ckpt = 'model/config/clip'
    resume = 'C:/Users/WangHL/Desktop/Heart_SegVol/SegVol/medsam_model_e500.pth'
    sam_model = sam_model_registry['vit']()
    segvol_model = SegVol(
                        image_encoder=sam_model.image_encoder, 
                        mask_decoder=sam_model.mask_decoder,
                        prompt_encoder=sam_model.prompt_encoder,
                        clip_ckpt=clip_ckpt,
                        roi_size=(32,256,256),
                        patch_size=(4,16,16),
                        test_mode=True,
                        )
    segvol_model = torch.nn.DataParallel(segvol_model)
    segvol_model.eval()
    # load param
    if os.path.isfile(resume):
        ## Map model to be loaded to specified single GPU
        loc = 'cpu'
        checkpoint = torch.load(resume, map_location=loc)
        segvol_model.load_state_dict(checkpoint['model'], strict=True)
        print("loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
    print('model build done!')
    return segvol_model

@st.cache_data
def inference_case(_image, _image_zoom_out, _point_prompt, text_prompt, _box_prompt):
    # seg config
    args = set_parse()
    args.use_zoom_in = False
    args.use_text_prompt = text_prompt is not None
    args.use_box_prompt = _box_prompt is not None
    args.use_point_prompt = _point_prompt is not None

    segvol_model = build_model()

    # run inference
    logits = zoom_in_zoom_out(
        args, segvol_model, 
        _image.unsqueeze(0), _image_zoom_out.unsqueeze(0), 
        text_prompt, _point_prompt, _box_prompt)
    print(logits.shape)
    resize_transform = transforms.Compose([
        transforms.AddChannel(),
        transforms.Resize((325,325,325), mode='trilinear')
    ]
    )
    logits_resize = resize_transform(logits)[0]
    return (torch.sigmoid(logits_resize) > 0.5).int().numpy(), (torch.sigmoid(logits) > 0.5).int().numpy()
    
