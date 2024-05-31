# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# from streamlit_image_coordinates import streamlit_image_coordinates


# from model.data_process.demo_data_process import process_ct_gt
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
# import monai.transforms as transforms
# from utils import show_points, make_fig, reflect_points_into_model, initial_rectangle, reflect_json_data_to_3D_box, reflect_box_into_model, run
# import nibabel as nib
# import tempfile

# print('script run')

# #############################################
# # init session_state
# if 'option' not in  st.session_state:
#     st.session_state.option = None
# if 'text_prompt' not in st.session_state:
#     st.session_state.text_prompt = None

# if 'reset_demo_case' not in st.session_state:
#     st.session_state.reset_demo_case = False

# if 'preds_3D' not in st.session_state:
#     st.session_state.preds_3D = None
#     st.session_state.preds_3D_ori = None
#     st.session_state.semantic_res = None

# if 'data_item' not in st.session_state:
#     st.session_state.data_item = None

# if 'points' not in st.session_state:
#     st.session_state.points = []

# if 'use_text_prompt' not in st.session_state:
#     st.session_state.use_text_prompt = False

# if 'use_point_prompt' not in st.session_state:
#     st.session_state.use_point_prompt = False

# if 'use_box_prompt' not in st.session_state:
#     st.session_state.use_box_prompt = False

# if 'rectangle_3Dbox' not in st.session_state:
#     st.session_state.rectangle_3Dbox = [0,0,0,0,0,0]

# if 'irregular_box' not in st.session_state:
#     st.session_state.irregular_box = False

# if 'running' not in st.session_state:
#     st.session_state.running = False

# if 'transparency' not in st.session_state:
#     st.session_state.transparency = 0.25

# case_list = [
#     './model/asset/54.nii.gz',
#     './model/asset/55.nii.gz',
#     './model/asset/la_003.nii.gz',
#     './model/asset/la_004.nii.gz'
# ]

# #############################################

# #############################################
# # reset functions
# def clear_prompts():
#     st.session_state.points = []
#     st.session_state.rectangle_3Dbox = [0,0,0,0,0,0]

# def reset_demo_case():
#     st.session_state.data_item = None
#     st.session_state.reset_demo_case = True
#     clear_prompts()

# def clear_file():
#     st.session_state.option = None
#     process_ct_gt.clear()
#     reset_demo_case()
#     clear_prompts()

# #############################################

# st.image(Image.open('model/asset/method_back.jpg'), use_column_width=True)

# github_col, arxive_col = st.columns(2)

# with github_col:
#     st.write('GitHub repo:https://github.com/BAAI-DCAI/SegVol')

# with arxive_col:
#     st.write('Paper:https://arxiv.org/abs/2311.13385')


# # modify demo case here
# demo_type = st.radio(
#         "Demo case source",
#         ["Select", "Upload"],
#         on_change=clear_file
#     )

# if demo_type=="Select":
#     uploaded_file = st.selectbox(
#         "Select a demo case",
#         case_list,
#         index=None,
#         placeholder="Select a demo case...",
#         on_change=reset_demo_case
#     )
# else:
#     uploaded_file = st.file_uploader("Upload demo case(nii.gz)", type='nii.gz', on_change=reset_demo_case)

# st.session_state.option = uploaded_file

# if  st.session_state.option is not None and \
#     st.session_state.reset_demo_case or (st.session_state.data_item is None and st.session_state.option is not None):

#     st.session_state.data_item = process_ct_gt(st.session_state.option)
#     st.session_state.reset_demo_case = False
#     st.session_state.preds_3D = None
#     st.session_state.preds_3D_ori = None
#     st.session_state.semantic_res = None

# prompt_col1, prompt_col2 = st.columns(2)

# with prompt_col1:
#     st.session_state.use_text_prompt = st.toggle('Sematic prompt')
#     text_prompt_type = st.radio(
#         "Sematic prompt type",
#         ["Predefined", "Custom"],
#         disabled=(not st.session_state.use_text_prompt)
#     )
#     if text_prompt_type == "Predefined":
#         pre_text = st.selectbox(
#             "Predefined anatomical category:",
#             ['liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 'inferior vena cava', 'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus', 'stomach', 'duodenum', 'left kidney'],
#             index=None,
#             disabled=(not st.session_state.use_text_prompt)
#         )
#     else:
#         pre_text = st.text_input('Enter an Anatomical word or phrase:', None, max_chars=20,
#                                                      disabled=(not st.session_state.use_text_prompt))
#     if pre_text is None or len(pre_text) > 0:
#         st.session_state.text_prompt = pre_text
#     else:
#         st.session_state.text_prompt = None


# with prompt_col2:
#     spatial_prompt_on = st.toggle('Spatial prompt', on_change=clear_prompts)
#     spatial_prompt = st.radio(
#         "Spatial prompt type",
#         ["Point prompt", "Box prompt"],
#         on_change=clear_prompts,
#         disabled=(not spatial_prompt_on))
#     st.session_state.enforce_zoom = st.checkbox('Enforce zoom-out-zoom-in')

# if spatial_prompt == "Point prompt":
#     st.session_state.use_point_prompt = True
#     st.session_state.use_box_prompt = False
# elif spatial_prompt == "Box prompt":
#     st.session_state.use_box_prompt = True
#     st.session_state.use_point_prompt = False
# else:
#     st.session_state.use_point_prompt = False
#     st.session_state.use_box_prompt = False

# if not spatial_prompt_on:
#     st.session_state.use_point_prompt = False
#     st.session_state.use_box_prompt = False

# if not st.session_state.use_text_prompt:
#     st.session_state.text_prompt = None

# if st.session_state.option is None:
#     st.write('please select demo case first')
# else:
#     image_3D = st.session_state.data_item['z_image'][0].numpy()
#     col_control1, col_control2 = st.columns(2)

#     with col_control1:
#         selected_index_z = st.slider('X-Y view', 0, image_3D.shape[0] - 1, 162, key='xy', disabled=st.session_state.running)

#     with col_control2:
#         selected_index_y = st.slider('X-Z view', 0, image_3D.shape[1] - 1, 162, key='xz', disabled=st.session_state.running)
#         if st.session_state.use_box_prompt:
#             top, bottom = st.select_slider(
#                 'Top and bottom of box',
#                 options=range(0, 325),
#                 value=(0, 324), 
#                 disabled=st.session_state.running
#             )
#             st.session_state.rectangle_3Dbox[0] = top
#             st.session_state.rectangle_3Dbox[3] = bottom
#     col_image1, col_image2 = st.columns(2)

#     if st.session_state.preds_3D is not None:
#         st.session_state.transparency = st.slider('Mask opacity', 0.0, 1.0, 0.25, disabled=st.session_state.running)

#     with col_image1:
        
#         image_z_array = image_3D[selected_index_z]

#         preds_z_array = None
#         if st.session_state.preds_3D is not None:
#             preds_z_array = st.session_state.preds_3D[selected_index_z]
            
#         image_z = make_fig(image_z_array, preds_z_array, st.session_state.points, selected_index_z, 'xy')
        
        
#         if st.session_state.use_point_prompt:
#             value_xy = streamlit_image_coordinates(image_z, width=325)
            
#             if value_xy is not None:
#                 point_ax_xy = (selected_index_z, value_xy['y'], value_xy['x'])
#                 if len(st.session_state.points) >= 3:
#                     st.warning('Max point num is 3', icon="⚠️")
#                 elif point_ax_xy not in st.session_state.points:
#                     st.session_state.points.append(point_ax_xy)
#                     print('point_ax_xy add rerun')
#                     st.rerun()
#         elif st.session_state.use_box_prompt:
#             canvas_result_xy = st_canvas(
#                 fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#                 stroke_width=3,
#                 stroke_color='#2909F1',
#                 background_image=image_z,
#                 update_streamlit=True,
#                 height=325,
#                 width=325,
#                 drawing_mode='transform',
#                 point_display_radius=0,
#                 key="canvas_xy",
#                 initial_drawing=initial_rectangle,
#                 display_toolbar=True
#             )
#             try:
#                 print(canvas_result_xy.json_data['objects'][0]['angle'])
#                 if canvas_result_xy.json_data['objects'][0]['angle'] != 0:
#                     st.warning('Rotating is undefined behavior', icon="⚠️")
#                     st.session_state.irregular_box = True
#                 else:
#                     st.session_state.irregular_box = False
#                 reflect_json_data_to_3D_box(canvas_result_xy.json_data, view='xy')
#             except:
#                 print('exception')
#                 pass
#         else:
#             st.image(image_z, use_column_width=False)

#     with col_image2:
#         image_y_array = image_3D[:, selected_index_y, :]
        
#         preds_y_array = None
#         if st.session_state.preds_3D is not None:
#             preds_y_array = st.session_state.preds_3D[:, selected_index_y, :]
        
#         image_y = make_fig(image_y_array, preds_y_array, st.session_state.points, selected_index_y, 'xz')
        
#         if st.session_state.use_point_prompt:
#             value_yz = streamlit_image_coordinates(image_y, width=325)
            
#             if value_yz is not None:
#                 point_ax_xz = (value_yz['y'], selected_index_y, value_yz['x'])
#                 if len(st.session_state.points) >= 3:
#                     st.warning('Max point num is 3', icon="⚠️")
#                 elif point_ax_xz not in st.session_state.points:
#                     st.session_state.points.append(point_ax_xz)
#                     print('point_ax_xz add rerun')
#                     st.rerun()
#         elif st.session_state.use_box_prompt:
#             if st.session_state.rectangle_3Dbox[1] <= selected_index_y and selected_index_y <= st.session_state.rectangle_3Dbox[4]:
#                 draw = ImageDraw.Draw(image_y)
#                 #rectangle xz view (upper-left and lower-right)
#                 rectangle_coords = [(st.session_state.rectangle_3Dbox[2], st.session_state.rectangle_3Dbox[0]),
#                                     (st.session_state.rectangle_3Dbox[5], st.session_state.rectangle_3Dbox[3])]
#                 # Draw the rectangle on the image
#                 draw.rectangle(rectangle_coords, outline='#2909F1', width=3)
#             st.image(image_y, use_column_width=False)
#         else:
#             st.image(image_y, use_column_width=False)

# if st.session_state.semantic_res is not None:
#     st.write('The mask is ', st.session_state.semantic_res)

# col1, col2, col3 = st.columns(3)

# with col1:
#     if st.button("Clear", use_container_width=True,
#                  disabled=(st.session_state.option is None or (len(st.session_state.points)==0 and not st.session_state.use_box_prompt and st.session_state.preds_3D is None))):
#         clear_prompts()
#         st.session_state.preds_3D = None
#         st.session_state.preds_3D_ori = None
#         st.rerun()

# with col2:
#     img_nii = None
#     if st.session_state.preds_3D_ori is not None and st.session_state.data_item is not None:
#         meta_dict = st.session_state.data_item['meta']
#         foreground_start_coord = st.session_state.data_item['foreground_start_coord']
#         foreground_end_coord = st.session_state.data_item['foreground_end_coord']
#         original_shape = st.session_state.data_item['ori_shape']
#         pred_array = st.session_state.preds_3D_ori

#         original_array = np.zeros(original_shape)
#         print(original_array.shape, pred_array.shape)
#         original_array[foreground_start_coord[0]:foreground_end_coord[0], 
#                     foreground_start_coord[1]:foreground_end_coord[1], 
#                     foreground_start_coord[2]:foreground_end_coord[2]] = pred_array

#         original_array = original_array.transpose(2, 1, 0)
#         img_nii = nib.Nifti1Image(original_array, affine=meta_dict['affine'])

#         with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmpfile:
#             nib.save(img_nii, tmpfile.name)
#             with open(tmpfile.name, "rb") as f:
#                 bytes_data = f.read()
#                 st.download_button(
#                     label="Download result(.nii.gz)",
#                     data=bytes_data,
#                     file_name="segvol_preds.nii.gz",
#                     mime="application/octet-stream",
#                     disabled=img_nii is None
#                 )
    

# with col3:
#     run_button_name = 'Run'if not st.session_state.running else 'Running'
#     if st.button(run_button_name, type="primary", use_container_width=True,
#             disabled=(
#                 st.session_state.data_item is None or
#                 (st.session_state.text_prompt is None and len(st.session_state.points) == 0 and st.session_state.use_box_prompt is False) or 
#                 st.session_state.irregular_box or 
#                 st.session_state.running
#                 )):
#         st.session_state.running = True
#         st.rerun()

# if st.session_state.running:
#     st.session_state.running = False
#     with st.status("Running...", expanded=False) as status:
#         run()
#     st.rerun()
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
from model.data_process.demo_data_process import process_ct_gt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import monai.transforms as transforms
from utils import show_points, make_fig, reflect_points_into_model, initial_rectangle, reflect_json_data_to_3D_box, reflect_box_into_model, run
import nibabel as nib
import tempfile
import os

print('script run')

#############################################
# init session_state
if 'option' not in  st.session_state:
    st.session_state.option = None
if 'text_prompt' not in st.session_state:
    st.session_state.text_prompt = None
if 'reset_demo_case' not in st.session_state:
    st.session_state.reset_demo_case = False
if 'preds_3D' not in st.session_state:
    st.session_state.preds_3D = None
    st.session_state.preds_3D_ori = None
    st.session_state.semantic_res = None
if 'data_item' not in st.session_state:
    st.session_state.data_item = None
if 'points' not in st.session_state:
    st.session_state.points = []
if 'use_text_prompt' not in st.session_state:
    st.session_state.use_text_prompt = False
if 'use_point_prompt' not in st.session_state:
    st.session_state.use_point_prompt = False
if 'use_box_prompt' not in st.session_state:
    st.session_state.use_box_prompt = False
if 'rectangle_3Dbox' not in st.session_state:
    st.session_state.rectangle_3Dbox = [0,0,0,0,0,0]
if 'irregular_box' not in st.session_state:
    st.session_state.irregular_box = False
if 'running' not in st.session_state:
    st.session_state.running = False
if 'transparency' not in st.session_state:
    st.session_state.transparency = 0.25

case_list = [
    './model/asset/54.nii.gz',
    './model/asset/55.nii.gz',
    './model/asset/la_003.nii.gz',
    './model/asset/la_004.nii.gz'
]

#############################################
# reset functions
def clear_prompts():
    st.session_state.points = []
    st.session_state.rectangle_3Dbox = [0,0,0,0,0,0]

def reset_demo_case():
    st.session_state.data_item = None
    st.session_state.reset_demo_case = True
    clear_prompts()

def clear_file():
    st.session_state.option = None
    process_ct_gt.clear()
    reset_demo_case()
    clear_prompts()

#############################################

#st.image(Image.open('model/asset/method_back.jpg'), use_column_width=True)

github_col, arxive_col = st.columns(2)

# with github_col:
#     st.write('GitHub repo:https://github.com/BAAI-DCAI/SegVol')

# with arxive_col:
#     st.write('Paper:https://arxiv.org/abs/2311.13385')

# modify demo case here
demo_type = st.radio(
        "Demo case source",
        ["Select", "Upload"],
        on_change=clear_file
    )

if demo_type == "Select":
    uploaded_file = st.selectbox(
        "Select a demo case",
        case_list,
        index=None,
        placeholder="Select a demo case...",
        on_change=reset_demo_case
    )
else:
    uploaded_file = st.file_uploader("Upload demo case(nii.gz)", type='nii.gz', on_change=reset_demo_case)

st.session_state.option = uploaded_file

if st.session_state.option is not None and \
   (st.session_state.reset_demo_case or st.session_state.data_item is None):
    st.session_state.data_item = process_ct_gt(st.session_state.option)
    st.session_state.reset_demo_case = False
    st.session_state.preds_3D = None
    st.session_state.preds_3D_ori = None
    st.session_state.semantic_res = None

prompt_col1, prompt_col2 = st.columns(2)

with prompt_col1:
    st.session_state.use_text_prompt = st.checkbox('Sematic prompt')
    text_prompt_type = st.radio(
        "Sematic prompt type",
        ["Predefined", "Custom"],
        disabled=(not st.session_state.use_text_prompt)
    )
    if text_prompt_type == "Predefined":
        pre_text = st.selectbox(
            "Predefined anatomical category:",
            [
                "ascending aorta",
                "descending aorta",
                "inferior vena cava",
                "left atrium",
                "left ventricle",
                "pulmonary artery",
                "pulmonary veins",
                "right atrium",
                "right ventricle",
                "superior vena cava"
            ],
            index=None,
            disabled=(not st.session_state.use_text_prompt)
        )
    else:
        pre_text = st.text_input('Enter an Anatomical word or phrase:', None, max_chars=20,
                                                     disabled=(not st.session_state.use_text_prompt))
    st.session_state.text_prompt = pre_text

with prompt_col2:
    spatial_prompt_on = st.checkbox('Spatial prompt', on_change=clear_prompts)
    spatial_prompt = st.radio(
        "Spatial prompt type",
        ["Point prompt", "Box prompt"],
        on_change=clear_prompts,
        disabled=(not spatial_prompt_on))
    st.session_state.enforce_zoom = st.checkbox('Enforce zoom-out-zoom-in')

if spatial_prompt == "Point prompt":
    st.session_state.use_point_prompt = True
    st.session_state.use_box_prompt = False
elif spatial_prompt == "Box prompt":
    st.session_state.use_box_prompt = True
    st.session_state.use_point_prompt = False
else:
    st.session_state.use_point_prompt = False
    st.session_state.use_box_prompt = False

if not spatial_prompt_on:
    st.session_state.use_point_prompt = False
    st.session_state.use_box_prompt = False

if not st.session_state.use_text_prompt:
    st.session_state.text_prompt = None

if st.session_state.option is None:
    st.write('please select demo case first')
else:
    image_3D = st.session_state.data_item['z_image'][0].numpy()
    col_control1, col_control2 = st.columns(2)

    with col_control1:
        selected_index_z = st.slider('X-Y view', 0, image_3D.shape[0] - 1, 162, key='xy', disabled=st.session_state.running)

    with col_control2:
        selected_index_y = st.slider('X-Z view', 0, image_3D.shape[1] - 1, 162, key='xz', disabled=st.session_state.running)
        if st.session_state.use_box_prompt:
            top, bottom = st.select_slider(
                'Top and bottom of box',
                options=range(0, 325),
                value=(0, 324), 
                disabled=st.session_state.running
            )
            st.session_state.rectangle_3Dbox[0] = top
            st.session_state.rectangle_3Dbox[3] = bottom

    col_image1, col_image2 = st.columns(2)

    if st.session_state.preds_3D is not None:
        st.session_state.transparency = st.slider('Mask opacity', 0.0, 1.0, 0.25, disabled=st.session_state.running)

    with col_image1:
        image_z_array = image_3D[selected_index_z]
        preds_z_array = None
        if st.session_state.preds_3D is not None:
            preds_z_array = st.session_state.preds_3D[selected_index_z]
        image_z = make_fig(image_z_array, preds_z_array, st.session_state.points, selected_index_z, 'xy')
        
        if st.session_state.use_point_prompt:
            value_xy = streamlit_image_coordinates(image_z, width=325)
            if value_xy is not None:
                point_ax_xy = (selected_index_z, value_xy['y'], value_xy['x'])
                if len(st.session_state.points) >= 3:
                    st.warning('Max point num is 3', icon="⚠️")
                elif point_ax_xy not in st.session_state.points:
                    st.session_state.points.append(point_ax_xy)
                    print('point_ax_xy add rerun')
                    st.rerun()
        elif st.session_state.use_box_prompt:
            canvas_result_xy = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=3,
                stroke_color='#2909F1',
                background_image=image_z,
                update_streamlit=True,
                height=325,
                width=325,
                drawing_mode='transform',
                point_display_radius=0,
                key="canvas_xy",
                initial_drawing=initial_rectangle,
                display_toolbar=True
            )
            try:
                print(canvas_result_xy.json_data['objects'][0]['angle'])
                if canvas_result_xy.json_data['objects'][0]['angle'] != 0:
                    st.warning('Rotating is undefined behavior', icon="⚠️")
                    st.session_state.irregular_box = True
                else:
                    st.session_state.irregular_box = False
                reflect_json_data_to_3D_box(canvas_result_xy.json_data, view='xy')
            except:
                print('exception')
                pass
        else:
            st.image(image_z, use_column_width=False)

    with col_image2:
        image_y_array = image_3D[:, selected_index_y, :]
        preds_y_array = None
        if st.session_state.preds_3D is not None:
            preds_y_array = st.session_state.preds_3D[:, selected_index_y, :]
        image_y = make_fig(image_y_array, preds_y_array, st.session_state.points, selected_index_y, 'xz')
        
        if st.session_state.use_point_prompt:
            value_yz = streamlit_image_coordinates(image_y, width=325)
            if value_yz is not None:
                point_ax_xz = (value_yz['y'], selected_index_y, value_yz['x'])
                if len(st.session_state.points) >= 3:
                    st.warning('Max point num is 3', icon="⚠️")
                elif point_ax_xz not in st.session_state.points:
                    st.session_state.points.append(point_ax_xz)
                    print('point_ax_xz add rerun')
                    st.rerun()
        elif st.session_state.use_box_prompt:
            if st.session_state.rectangle_3Dbox[1] <= selected_index_y <= st.session_state.rectangle_3Dbox[4]:
                draw = ImageDraw.Draw(image_y)
                # rectangle xz view (upper-left and lower-right)
                rectangle_coords = [(st.session_state.rectangle_3Dbox[2], st.session_state.rectangle_3Dbox[0]),
                                    (st.session_state.rectangle_3Dbox[5], st.session_state.rectangle_3Dbox[3])]
                # Draw the rectangle on the image
                draw.rectangle(rectangle_coords, outline='#2909F1', width=3)
            st.image(image_y, use_column_width=False)
        else:
            st.image(image_y, use_column_width=False)

if st.session_state.semantic_res is not None:
    st.write('The mask is ', st.session_state.semantic_res)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Clear", use_container_width=True,
                 disabled=(st.session_state.option is None or (len(st.session_state.points) == 0 and not st.session_state.use_box_prompt and st.session_state.preds_3D is None))):
        clear_prompts()
        st.session_state.preds_3D = None
        st.session_state.preds_3D_ori = None
        st.rerun()

with col2:
    img_nii = None
    if st.session_state.preds_3D_ori is not None and st.session_state.data_item is not None:
        meta_dict = st.session_state.data_item['meta']
        foreground_start_coord = st.session_state.data_item['foreground_start_coord']
        foreground_end_coord = st.session_state.data_item['foreground_end_coord']
        original_shape = st.session_state.data_item['ori_shape']
        pred_array = st.session_state.preds_3D_ori

        original_array = np.zeros(original_shape)
        print(original_array.shape, pred_array.shape)
        original_array[foreground_start_coord[0]:foreground_end_coord[0], 
                       foreground_start_coord[1]:foreground_end_coord[1], 
                       foreground_start_coord[2]:foreground_end_coord[2]] = pred_array

        original_array = original_array.transpose(2, 1, 0)
        img_nii = nib.Nifti1Image(original_array, affine=meta_dict['affine'])

        temp_dir = "C:/Users/WangHL/Desktop/Heart_SegVol/TempDir"
        os.makedirs(temp_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", dir=temp_dir, delete=False) as tmpfile:
            nib.save(img_nii, tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                bytes_data = f.read()
                st.download_button(
                    label="Download result(.nii.gz)",
                    data=bytes_data,
                    file_name="segvol_preds.nii.gz",
                    mime="application/octet-stream",
                    disabled=img_nii is None
                )
        os.remove(tmpfile.name)

with col3:
    run_button_name = 'Run' if not st.session_state.running else 'Running'
    if st.button(run_button_name, type="primary", use_container_width=True,
                 disabled=(st.session_state.data_item is None or
                           (st.session_state.text_prompt is None and len(st.session_state.points) == 0 and not st.session_state.use_box_prompt) or 
                           st.session_state.irregular_box or 
                           st.session_state.running)):
        st.session_state.running = True
        st.rerun()

if st.session_state.running:
    st.session_state.running = False
    with st.spinner("Running..."):
        run()
    st.rerun()
