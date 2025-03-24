### 安装本项目

```bash
git clone https://github.com/wangbingxun02/ctv_challenge.git
```

```bash
cd ctv_challenge
pip install -e .
```

创建nnUNet的数据目录

```
nnUNetFrame
|---nnUNet_raw
|---nnUNet_preprocessed
|---nnUNet_results
```

创建完成后，需要设置路径:

```bash
export nnUNet_raw="raw的路径"
export nnUNet_preprocessed=" preprocessed 的路径" 
export nnUNet_results="results 的路径"
```


### 数据划分
将原始数据集转换成符合nnUNet的格式，并存入nnUNetFrame/nnUNet_raw 文件夹中。代码见Dataset009_ctv_challenge.py

**注意：**
原始数据集的train、val、test文件夹中的样例编号均为从p_0开始，所以需要提前将val和test中的编号改为100之后，以避免数据因编号一致造成的冲突。如：  
validation文件夹中， p_0 -> p_100 ，p_1 -> p_101 ...  
test文件夹中， p_0 -> p_120 ，p_1 -> p_121...  

运行Dataset009_ctv_challenge.py来划分数据集：
```bash
Python3  Dataset009_ctv_challenge.py -i  原始数据集的路径 -d 设置的此dataset的ID(0-999)
```
转换后，目录如下：
```
nnUNetFrame
|---nnUNet_raw
|   |---Dataset009_ctv_challenge
|   |   |---imagesTr
|   |   |---imagesTs
|   |   |---labelsTr
|   |   |---dataset.json
|---nnUNet_preprocessed
|---nnUNet_results
```
该数据集用于**第一阶段**的训练，生成伪标签。

### 数据预处理

```bash
nnUNetv2_plan_and_preprocess -d 设置的此dataset的ID --verify_dataset_integrity
```

### 训练与推理
本方法采用nnUNetV2框架中的nnUNet-3d（fullres）模型作为分割模型，并使用五折交叉验证，因此，需要训练五次模型。
```bash
nnUNetv2_train 设置的此dataset的ID 3d_fullres 0
nnUNetv2_train 设置的此dataset的ID 3d_fullres 1
nnUNetv2_train 设置的此dataset的ID 3d_fullres 2
nnUNetv2_train 设置的此dataset的ID 3d_fullres 3
nnUNetv2_train 设置的此dataset的ID 3d_fullres 4
```

训练完5折后进行推理:
```bash
nnUNetv2_predict -i /staff/wangbingxun/projects/nnUnet/nnUNetFrame/DATASET/nnUNet_raw/Dataset012_ctv_challenge/imagesTs（示例路径） -o 自己设置的存储推理结果的文件夹 -d 设置的此dataset的ID -c 3d_fullres
```

### 第二阶段训练
* 将推理的结果作为伪标签，进行第二阶段的训练。  
具体来说，再次运行Dataset009_ctv_challenge.py，为新数据集设置一个新的ID得到一个新的数据集，在此数据集中，将imagesTs文件夹中的样本复制到imagesTr文件夹中，将第一阶段训练中推理的结果复制到labelsTr文件夹中。  
* 再次运行数据预处理、训练与推理指令，得到的推理结果即为第二阶段的结果。

### 后处理
对第二阶段的结果进行后处理。代码见post_processing.py，在文件中修改路径后，运行即可。
```bash
python3 post_processing.py
```
得到的结果即为最终的结果。
