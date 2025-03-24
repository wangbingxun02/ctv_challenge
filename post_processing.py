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

        # **（第三步）若包含两个即以上区域，只保留最大的区域** 
        labeled_slice_2, num_features_2 = ndimage.label(cleaned_mask)  # 连通区域分析
        if num_features_2 > 1:
            # 计算每个连通区域的像素数
            region_sizes = [np.sum(labeled_slice_2 == label) for label in range(1, num_features_2 + 1)]
            # 找到最大的区域的索引
            max_region_index = np.argmax(region_sizes) + 1  # 加1是因为索引从0开始
            # 创建一个空白 mask
            cleaned_mask_2 = np.zeros_like(cleaned_mask, dtype=np.int8)    
            # 只保留最大的区域
            # cleaned_mask_2[labeled_slice_2 == max_region_index] = 1  # 只保留最大的区域
            cleaned_mask = cleaned_mask_2


        # if num_features > 1:
        #     # 计算每个连通区域的像素数
        #     region_sizes = [np.sum(labeled_slice == label) for label in range(1, num_features + 1)]
        #     # 找到最大的区域的索引
        #     max_region_index = np.argmax(region_sizes) + 1  # 加1是因为索引从0开始
        #     # 创建一个空白 mask
        #     cleaned_mask = np.zeros_like(cleaned_mask, dtype=np.int8)
        #     # 只保留最大的区域
        #     cleaned_mask[labeled_slice == max_region_index] = 1  # 只保留最大的区域



        # 更新处理后的数据
        processed_data[:, :, z] = cleaned_mask  
        print(f"数据类型: {processed_data.dtype}")


    # 3. 保存处理后的 .nii.gz 文件
    processed_nii = nib.Nifti1Image(processed_data, nii.affine, nii.header, dtype=np.int8)
    nib.save(processed_nii, output_path)
    print(f"处理完成，保存至: {output_path}")

def let_me_know_region_size(input_path):
    """
    遍历.nii.gz 文件的每个切片，打印每个 mask 区域的像素数。
    参数:
    - input_path: str, 输入的.nii.gz label 文件路径"
    """
    # 1. 读取 label 文件
    nii = nib.load(input_path)
    data = nii.get_fdata()  # 获取 3D numpy 数组
    # 读取数据类型
    data_type = data.dtype
    # 打印数据类型
    print(f"数据类型: {data_type}")
    processed_data = np.zeros_like(data, dtype=np.int8)  # 初始化空白 mask
    # 打印数据类型
    print(f"数据类型: {processed_data.dtype}")

    # 2. 遍历每个切片（假设是 z 轴方向）
    for z in range(data.shape[2]):
        slice_mask = (data[:, :, z] == 1)  # 获取当前切片中的 mask

        # 3. 进行连通区域分析，识别不同的 mask 组件
        labeled_slice, num_features = ndimage.label(slice_mask)
        # 4. 遍历每个连通区域，计算其像素数
        for label in range(1, num_features + 1):
            region_size = np.sum(labeled_slice == label)  # 计算该区域的像素数量
            print(f"在切片 {z + 1} 的 mask 区域 {label} 有 {region_size} 个像素。")
            # print(labeled_slice)
            # print(label)
    
    

# # 示例用法
# input_label_file = "/staff/wangbingxun/new_nnUNet/output/Dataset017/nnunet_chk_200_region_delete/p_416.nii.gz"  # 替换为你的输入文件路径
# # output_file = "/staff/wangbingxun/new_nnUNet/output/Dataset018/nnunet_chk_best_0-403_with_pseudo_label_pp200_max_region"  # 替换为你的输出文件路径
# # process_label_file(input_label_file, output_file, 200)
# let_me_know_region_size(input_label_file)

# 示例用法
input_label_file = "/staff/wangbingxun/new_nnUNet/output/Dataset018/nnunet_test"   # 替换为你的输入文件路径
output_label_file = "/staff/wangbingxun/new_nnUNet/output/Dataset018/nnunet_test_pp300_delete_region" # 替换为你的输出文件路径
threshold_value = 300  # 设定阈值，比如小于100个像素的mask区域会被删除

# 将文件夹中的.nii.gz文件进行处理

for file in os.listdir(input_label_file):
    if file.endswith(".nii.gz"):
        input_file = os.path.join(input_label_file, file)
        print(input_file)
        output_file = os.path.join(output_label_file, file)
        process_label_file(input_file, output_file, threshold_value)