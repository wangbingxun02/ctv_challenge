import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import os

def process_label_file(input_path, output_path, threshold):
    """
    处理 .nii.gz 文件：
    1. 填充 mask 内部空洞。
    2. 移除面积小于阈值的连通区域。
    3. 保存为 .nii.gz 格式。

    参数:
    - input_path: str, 输入的 .nii.gz label 文件路径
    - output_path: str, 处理后的 .nii.gz 文件保存路径
    - threshold: int, 低于该像素数的 mask 区域将被删除
    """
    # 1. 读取 label 文件
    nii = nib.load(input_path)
    data = nii.get_fdata()  # 获取 3D numpy 数组

    processed_data = np.zeros_like(data, dtype=np.int8)  # 初始化空白 mask
    
    # 2. 遍历每个切片
    for z in range(data.shape[2]):  
        slice_mask = (data[:, :, z] == 1)  # 仅选取 mask 部分

        if np.sum(slice_mask) == 0:  # 当前切片没有 mask，则跳过
            continue  

        # **（第一步）填充内部空洞**
        filled_mask = ndimage.binary_fill_holes(slice_mask).astype(np.int8)

        # **（第二步）移除小的连通区域,只保留大于阈值的区域**
        labeled_slice, num_features = ndimage.label(filled_mask)  # 连通区域分析

        # 创建一个空白 mask
        cleaned_mask = np.zeros_like(filled_mask, dtype=np.int8)
        

        for label in range(1, num_features + 1):
            region_size = np.sum(labeled_slice == label)  # 计算连通区域的像素数
            print(f"Slice {z}: Region {label}, Size {region_size}")  # 调试输出
            
            if region_size > threshold + 1:
                cleaned_mask[labeled_slice == label] = 1  # 只保留大于阈值的区域

        # **（第三步）若切片中仍然包含多连通域，则将这些多连通域都去除** 
        labeled_slice_2, num_features_2 = ndimage.label(cleaned_mask)  # 连通区域分析
        if num_features_2 > 1:
            # 创建一个空白 mask
            cleaned_mask_2 = np.zeros_like(cleaned_mask, dtype=np.int8)    
            # 去除多联通域
            cleaned_mask = cleaned_mask_2


        # 更新处理后的数据
        processed_data[:, :, z] = cleaned_mask  
        # print(f"数据类型: {processed_data.dtype}")


    # 3. 保存处理后的 .nii.gz 文件
    processed_nii = nib.Nifti1Image(processed_data, nii.affine, nii.header, dtype=np.int8)
    nib.save(processed_nii, output_path)
    print(f"处理完成，保存至: {output_path}")

    
    


# 示例用法
input_label_file = " "   # 替换为你的输入文件路径
output_label_file = " " # 替换为你的输出文件路径
threshold_value = 300  # 设定阈值

# 将文件夹中的.nii.gz文件进行处理

for file in os.listdir(input_label_file):
    if file.endswith(".nii.gz"):
        input_file = os.path.join(input_label_file, file)
        print(input_file)
        output_file = os.path.join(output_label_file, file)
        process_label_file(input_file, output_file, threshold_value)