import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import torch
import streamlit as st
from model.inference_cpu import inference_case

initial_rectangle = {
    "version": "4.4.0",
    'objects': [
        {
            "type": "rect",
            "version": "4.4.0",
            "originX": "left",
            "originY": "top",
            "left": 50,
            "top": 50,
            "width": 100,
            "height": 100,
            'fill': 'rgba(255, 165, 0, 0.3)', 
            'stroke': '#2909F1', 
            'strokeWidth': 3, 
            'strokeDashArray': None, 
            'strokeLineCap': 'butt', 
            'strokeDashOffset': 0, 
            'strokeLineJoin': 'miter', 
            'strokeUniform': True, 
            'strokeMiterLimit': 4, 
            'scaleX': 1, 
            'scaleY': 1, 
            'angle': 0, 
            'flipX': False, 
            'flipY': False, 
            'opacity': 1, 
            'shadow': None, 
            'visible': True, 
            'backgroundColor': '', 
            'fillRule': 
            'nonzero', 
            'paintFirst': 
            'fill', 
            'globalCompositeOperation': 'source-over', 
            'skewX': 0, 
            'skewY': 0, 
            'rx': 0, 
            'ry': 0
        }
    ]
}

def run():
    image = st.session_state.data_item["image"].float()
    image_zoom_out = st.session_state.data_item["zoom_out_image"].float()
    text_prompt = None
    point_prompt = None
    box_prompt = None
    if st.session_state.use_text_prompt:
        text_prompt = st.session_state.text_prompt
    if st.session_state.use_point_prompt and len(st.session_state.points) > 0:
        point_prompt = reflect_points_into_model(st.session_state.points)
    if st.session_state.use_box_prompt:
        box_prompt = reflect_box_into_model(st.session_state.rectangle_3Dbox)
    inference_case.clear()
    st.session_state.preds_3D, st.session_state.preds_3D_ori = inference_case(image, image_zoom_out, 
                                            text_prompt=text_prompt,
                                            _point_prompt=point_prompt,
                                            _box_prompt=box_prompt)

def reflect_box_into_model(box_3d):
    z1, y1, x1, z2, y2, x2 = box_3d
    x1_prompt = int(x1 * 256.0 / 325.0)
    y1_prompt = int(y1 * 256.0 / 325.0)
    z1_prompt = int(z1 * 32.0 / 325.0)
    x2_prompt = int(x2 * 256.0 / 325.0)
    y2_prompt = int(y2 * 256.0 / 325.0)
    z2_prompt = int(z2 * 32.0 / 325.0)
    return torch.tensor(np.array([z1_prompt, y1_prompt, x1_prompt, z2_prompt, y2_prompt, x2_prompt]))

def reflect_json_data_to_3D_box(json_data, view):
    if view == 'xy':
        st.session_state.rectangle_3Dbox[1] = json_data['objects'][0]['top']
        st.session_state.rectangle_3Dbox[2] = json_data['objects'][0]['left']
        st.session_state.rectangle_3Dbox[4] = json_data['objects'][0]['top'] + json_data['objects'][0]['height'] * json_data['objects'][0]['scaleY']
        st.session_state.rectangle_3Dbox[5] = json_data['objects'][0]['left'] + json_data['objects'][0]['width'] * json_data['objects'][0]['scaleX']
    print(st.session_state.rectangle_3Dbox)

def reflect_points_into_model(points):
    points_prompt_list = []
    for point in points:
        z, y, x = point
        x_prompt = int(x * 256.0 / 325.0)
        y_prompt = int(y * 256.0 / 325.0)
        z_prompt = int(z * 32.0 / 325.0)
        points_prompt_list.append([z_prompt, y_prompt, x_prompt])
    points_prompt = np.array(points_prompt_list)
    points_label = np.ones(points_prompt.shape[0])
    print(points_prompt, points_label)
    return (torch.tensor(points_prompt), torch.tensor(points_label))

def show_points(points_ax, points_label, ax):
    color = 'red' if points_label == 0 else 'blue'
    ax.scatter(points_ax[0], points_ax[1], c=color, marker='o', s=200)

def make_fig(image, preds, point_axs=None, current_idx=None, view=None):
    # Convert A to an image
    image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Create a yellow mask from B
    if preds is not None:
        mask = np.where(preds == 1, 255, 0).astype(np.uint8)
        mask = Image.merge("RGB", 
                           (Image.fromarray(mask), 
                            Image.fromarray(mask), 
                            Image.fromarray(np.zeros_like(mask, dtype=np.uint8))))

        # Overlay the mask on the image
        image = Image.blend(image.convert("RGB"), mask, alpha=st.session_state.transparency)
    
    if point_axs is not None:
        draw = ImageDraw.Draw(image)
        radius = 5
        for point in point_axs:
            z, y, x = point
            if view == 'xy' and z == current_idx:
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="blue")
            elif view == 'xz'and y == current_idx:
                draw.ellipse((x-radius, z-radius, x+radius, z+radius), fill="blue")
    return image