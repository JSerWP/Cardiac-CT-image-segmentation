import numpy as np
import monai.transforms as transforms
import streamlit as st
import tempfile

class MinMaxNormalization(transforms.Transform):
    def __call__(self, data):
        d = dict(data)
        k = "image"
        d[k] = d[k] - d[k].min()
        d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
        return d

class DimTranspose(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.swapaxes(d[key], -1, -3)
        return d

class ForegroundNormalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            d[key] = self.normalize(d[key])
        return d
    
    def normalize(self, ct_narray):
        ct_voxel_ndarray = ct_narray.copy()
        ct_voxel_ndarray = ct_voxel_ndarray.flatten()
        thred = np.mean(ct_voxel_ndarray)
        voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 00.05)
        mean = np.mean(voxel_filtered)
        std = np.std(voxel_filtered)
        ### transform ###
        ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
        ct_narray = (ct_narray - mean) / max(std, 1e-8)
        return ct_narray
    
@st.cache_data
def process_ct_gt(case_path, spatial_size=(32,256,256)):
    if case_path is None:
        return None
    print('Data preprocessing...')
    # transform
    img_loader = transforms.LoadImage(dtype=np.float32)
    transform = transforms.Compose(
        [
            # transforms.LoadImage(dtype=np.float32),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            ForegroundNormalization(keys=["image"]),
            DimTranspose(keys=["image"]),
            MinMaxNormalization(),
            # transforms.SpatialPadd(keys=["image"], spatial_size=spatial_size, mode='constant'),
            transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.ToTensord(keys=["image"]),
        ]
    )
    zoom_out_transform = transforms.Resized(keys=["image"], spatial_size=spatial_size, mode='nearest-exact')
    z_transform = transforms.Resized(keys=["image"], spatial_size=(325,325,325), mode='nearest-exact')
    ###
    item = {}
    # generate ct_voxel_ndarray
    if type(case_path) is str:
        ct_voxel_ndarray, meta_tensor_dict = img_loader(case_path)
    else:
        bytes_data = case_path.read()
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as tmp:
            tmp.write(bytes_data)
            tmp.seek(0)
            ct_voxel_ndarray, meta_tensor_dict = img_loader(tmp.name)

    ct_voxel_ndarray = np.array(ct_voxel_ndarray).squeeze()
    ct_voxel_ndarray = np.expand_dims(ct_voxel_ndarray, axis=0)
    item['image'] = ct_voxel_ndarray
    ori_shape = np.swapaxes(ct_voxel_ndarray, -1, -3).shape[1:]

    # transform
    item = transform(item)
    print(item.keys())
    item_zoom_out = zoom_out_transform(item)
    item['zoom_out_image'] = item_zoom_out['image']
    item['ori_shape'] = ori_shape

    item_z = z_transform(item)
    item['z_image'] = item_z['image']
    item['meta'] = meta_tensor_dict
    print('img - foreground img ', ori_shape, item['image'].shape)
    post_shape = list(item['image'].shape[1:])
    # if list(ori_shape) == :
    #     item['foreground_start_coord'] = [0,0,0]
    #     item['foreground_end_coord'] = list(ori_shape)
    for idx in range(len(ori_shape)):
        if ori_shape[idx] == post_shape[idx]:
            item['foreground_start_coord'][idx] = 0
            item['foreground_end_coord'][idx] = ori_shape[idx]
    print('inside process ', item['foreground_start_coord'], item['foreground_end_coord'])
    return item
