from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

import random
import glob

random.seed(10)




def convert_ctv_challenge(src_data_folder: str, dataset_id=9):

    

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    
    # 命名此数据集
    task_name = "ctv_challenge"

    foldername = "Dataset%03.0d_%s" % (dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)


    train_dir = join(src_data_folder, 'Training')
    test_dir = join(src_data_folder, 'Test')
    val_dir = join(src_data_folder, 'Validation')
    case_ids = subdirs(train_dir,join=False)
    case_ts = subdirs(test_dir,join=False)
    case_val = subdirs(val_dir,join=False)
    print(case_ids)
    patients_train = sorted(case_ids, key=lambda x: int(x.split('_')[1]))
    
    patients_test = sorted(case_ts)
    patients_val = sorted(case_val)
    num_training_cases = len(patients_train)

    for c in patients_train:
        shutil.copy(join(train_dir, c, "image.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(train_dir, c, "CTV.nii.gz"), join(labelstr, c + '.nii.gz'))
    
    # imagestr = 原始数据集的训练集+验证集
    # imagests = 原始数据集的测试集
    for c in patients_val:
        shutil.copy(join(val_dir, c, "image.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(val_dir, c, "CTV.nii.gz"), join(labelstr, c + '.nii.gz'))
    for c in patients_test:
        shutil.copy(join(test_dir, c, "image.nii.gz"), join(imagests, c + '_0000.nii.gz'))
        
    
    generate_dataset_json(
                          out_base, {0: "CT"},
                          labels = {
                              "background": 0,
                              "ctv": 1
                          },
                          file_ending=".nii.gz",
                          num_training_cases=num_training_cases,
                          )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=9, help="nnU-Net Dataset ID, default: 009"
    )

    
    args = parser.parse_args()
    print("Converting...")
    convert_ctv_challenge(args.input_folder, args.dataset_id)
    print("Done!")
