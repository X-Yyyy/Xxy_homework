#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os.path
import os

def get_data_from_img(img_path):
    """
    Args:
        img_path:根据”系统分隔符“分割路径
    Returns:
    """
    p = os.path.normpath(img_path)
    return p.split(os.sep)


def check_first_image(relative_name_img):
    s = relative_name_img.split('_')
    return int(s[0]) == 0

"""
step 1:
    dataset_type="rgb","leap_motion","tof"
"""
"""
step 2:
    dataset_path=r"D:/PyCharmresorce/Gesturedataset/dataset/"
    自动匹配dataset_path目录下名为：dataset_type的文件夹
    最终形式为：
        r"D:/PyCharmresorce/Gesturedataset/dataset/rgb_mini"
        r"D:/PyCharmresorce/Gesturedataset/dataset/rgb"
        r"D:/PyCharmresorce/Gesturedataset/dataset/leap_motion"
        r"D:/PyCharmresorce/Gesturedataset/dataset/tof"
"""

def main(train_flag=True,validation_flag=True,dataset_type="rgb_mini",dataset_path = r"D:/PyCharmresorce/Gesturedataset/dataset/"):
    train = train_flag
    # gen_file_type = './csv_dataset.txt'
    if validation_flag == False:
        l = sorted([os.path.abspath(os.path.join(dp, f)) for dp, dn, fn in os.walk(
            os.path.expanduser(dataset_path+dataset_type+"/train" if train else dataset_path+dataset_type+"/test")) for f in fn])
        if train:
            gen_file_type = './csv_train_dataset_'+str(dataset_type)+'.txt'
            print("Train Csv ...\n")
        else:
            gen_file_type = './csv_test_dataset_'+str(dataset_type)+'.txt'
            print("Test Csv ...\n")

    elif validation_flag == True:
        l = sorted([os.path.abspath(os.path.join(dp, f)) for dp, dn, fn in os.walk(
            os.path.expanduser(
                dataset_path+dataset_type+"/train" if train else dataset_path+dataset_type+"/validation"))
                    for f in fn])
        if train:
            gen_file_type = './csv_train_dataset_'+str(dataset_type)+'.txt'
            print("Train Csv ...")
        else:
            gen_file_type = './csv_validation_dataset_'+str(dataset_type)+'.txt'
            print("Validation Csv ...")
    else:
        l = sorted([os.path.abspath(os.path.join(dp, f)) for dp, dn, fn in os.walk(
            os.path.expanduser(
                dataset_path+dataset_type+"/train" if train else dataset_path+dataset_type+"/test"))
                    for f in fn])
        if train:
            gen_file_type = './csv_train_dataset_'+str(dataset_type)+'.txt'
            print("Train Csv ...")
        else:
            gen_file_type = './csv_test_dataset_'+str(dataset_type)+'.txt'
            print("Test Csv ...")
    # list_sessions, num_of_sessions = os.listdir("data"), len(os.listdir("data"))
    # list_gestures = ['g0', 'g01', 'g02', 'g02', 'g02']

    with open(gen_file_type, 'w', newline="") as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')

        file_writer.writerow(['img_path', 'session_id', 'gesture_id', 'record', 'mode', 'label', 'first'])
        for img in l:
            data = get_data_from_img(img)
            # ['D:', 'PyCharmresorce', 'Gesturedataset', 'dataset', 'rgb', 'validation', '031', 'g11', '02', 'rgb', '062_rgb.png']
            # ['D:', 'PyCharmresorce', 'Gesturedataset', 'dataset', 'rgb', 'validation', '031', 'g12_test', '00', 'rgb', '365_rgb.png']
            print(data)
            #
            if data[-5] == 'g12_test' or data[-4] == 'g12_test':
                continue
            first = check_first_image(data[-1])
            if data[-2] == 'rgb':
                file_writer.writerow(
                    [img, data[-5], data[-4], data[-3], format(data[-2]), data[-4][1:]
                        , first])
            else:
                file_writer.writerow(
                    [img, data[-6], data[-5], data[-4], "{}_{}".format(data[-3], data[-2]), data[-5][1:]
                        , first])

    print(gen_file_type + ' COMPLETED')

if __name__ == '__main__':
    """
    Step 1, Generate datasets csv files. 
    """
    # if train_flag=True,validation_flag=False , then train csv file
    main(train_flag=True,validation_flag=False)
    # if train_flag=False,validation_flag=False , then  test csv file
    main(train_flag=False,validation_flag=False)
    # if train_flag=False,validation_flag=True , then  validation csv file
    main(train_flag=False,validation_flag=True)





