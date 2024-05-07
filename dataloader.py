from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2 #pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
import numpy as np
import csv
import torch
import utilities
import PIL
import math
import gzip
import os
import tqdm
import matplotlib.pyplot as plt
# csv
# img_path,session_id,gesture_id,record,mode,label,first
import sys
sys.path.append("./")

class GesturesDataset(Dataset):
    """
    mode : leap_motion_tracking_data/depth_ir、depth_z(tof)
    """
    def __init__(self, model, csv_path, train=False, mode='depth_ir', rgb=False, data_agumentation=False, normalization_type=-1,
                 preprocessing_type=-1, transform_train=False, resize_dim=64, n_frames=30, tracking_data_mod=False):
        super().__init__()
        self.model = model
        self.csv_path = csv_path
        self.train = train
        self.mode = mode
        self.rgb = rgb
        self.data_augmentation = data_agumentation
        self.normalization_type = normalization_type
        self.preprocessing_type = preprocessing_type
        self.resize_dim = resize_dim
        self.n_frames = n_frames
        self.tracking_data_mod = tracking_data_mod
        # self.crop_limit = crop_limit

        if self.mode != 'leap_motion_tracking_data' and self.model != 'C3D':
            if transform_train:
                if self.mode != 'depth_z':
                    self.transforms = transforms.Compose([
                        utilities.Rescale(256),
                        utilities.RandomCrop(self.resize_dim), #在一个随机的位置进行裁剪
                        utilities.RandomFlip(),
                        transforms.ToTensor()
                    ])
                else:
                    self.transforms = transforms.Compose([
                        utilities.Rescale(256),
                        utilities.RandomCrop(self.resize_dim),
                        # utilities.RandomFlip(15),
                    ])

            else:
                if self.mode != 'depth_z':
                    self.transforms = transforms.Compose([
                        transforms.ToTensor() # 归一化到(0,1)
                    ])
                else:
                    self.transforms = None

        self.list_data = []
        # print(self.csv_path)
        with open(self.csv_path, 'r') as csv_in:
            reader = csv.reader(csv_in)
            self.list_of_rows_with_first_frame = [row for row in reader if row[4] == self.mode and row[6] == 'True']
            # list_of_rows_with_same_mode = list_of_rows_with_same_mode[1:]  # list_of_row[1]['mode']
            csv_in.seek(0)
            self.list_of_rows_with_same_mode = [row for row in reader if row[4] == self.mode]
        print("first frame length :{}".format(len(self.list_of_rows_with_first_frame)))
        # print(self.list_of_rows_with_first_frame)
        print("dataset total length :{}".format(len(self.list_of_rows_with_same_mode)))
        # print(self.list_of_rows_with_same_mode)

        len_dataset_per_session = len(set([int(x[1]) for x in self.list_of_rows_with_first_frame]))
        data_index_list = list(set([int(x[1]) for x in self.list_of_rows_with_first_frame]))
        start_index = data_index_list[0]
        if start_index != 0:
            for i in range(0,len(data_index_list)):
                data_index_list[i] = data_index_list[i]-start_index
        # print(data_index_list)
        # txt文件中数据的组数000，001，...，026
        print("group number: ")
        print(len_dataset_per_session)
        # train_len = round(80 * len_dataset_per_session / 100)
        train_len = round(100 * len_dataset_per_session / 100)
        # print("train_len:")
        # print(train_len)
        for i in range(len_dataset_per_session):
            if self.train and i < train_len:
                self.list_data += ([x for x in self.list_of_rows_with_first_frame if (int(x[1])) == i])
            elif not self.train and i < train_len:
                # print("no train..........................")
                self.list_data += ([x for x in self.list_of_rows_with_first_frame if (data_index_list[i]) == i])
        # print("list_data:")
        # print(len(self.list_data))
        if self.mode == 'leap_motion_tracking_data':
            self.list_data_correct = []
            self.list_records = []
            for i, ff in enumerate(self.list_data):
                list_of_img_of_same_record = [img[0] for img in self.list_of_rows_with_same_mode
                                              if img[1] == ff[1]  # sessione
                                              and img[2] == ff[2]  # gesture
                                              and img[3] == ff[3]]  # record

                center_of_list = math.floor(len(list_of_img_of_same_record) / 2)
                crop_limit = math.floor(self.n_frames / 2)
                start = center_of_list - crop_limit
                end = center_of_list + crop_limit
                list_of_img_of_same_record_cropped = list_of_img_of_same_record[
                                                     start: end + 1 if self.n_frames % 2 == 1 else end]
                valid = True
                for frame in list_of_img_of_same_record_cropped:
                    js, j = utilities.from_json_to_list(frame)
                    if not js:
                        valid = False
                        break
                if valid:
                    self.list_records.append(list_of_img_of_same_record_cropped)
                    self.list_data_correct.append(ff)
# json to list
            list_of_ready_records = []
            for i, record in enumerate(self.list_records):
                list_of_json_frame = []
                for js in record:
                    f_js, j = utilities.from_json_to_list(js)
                    if self.tracking_data_mod:
                        f_js = utilities.extract_features_tracking_data(f_js)
                    list_of_json_frame.append(f_js)
                list_of_ready_records.append(list_of_json_frame)


            if self.tracking_data_mod:
            # print('ok')
                list_of_ready_records_increased = []
                for record in list_of_ready_records:
                    record = utilities.increase_input_size_per_record(record)
                    list_of_ready_records_increased.append(record)
                list_of_ready_records = list_of_ready_records_increased

            self.list_data = []
            for i in range(len(self.list_data_correct)):
                self.list_data.append((list_of_ready_records[i], self.list_data_correct[i]))

# 归一化
        if normalization_type:

            list_clip_to_norm = []
            if not os.path.exists("mean_std_{}.npz".format(self.mode)):
                print('calculating mean and std...')
                if self.mode == 'leap_motion_tracking_data':

                    frames_stack = []
                    for record, info in self.list_data:
                        for frame in record:
                            frames_stack.append((frame))

                    self.mean = np.mean(frames_stack)
                    self.std = np.std(frames_stack)

                    np.savez("mean_std_{}.npz".format(self.mode), self.mean, self.std)
                    print('mean, std {} saved.'.format(self.mode))

                else:

                    for i, first_img in enumerate(tqdm.tqdm(self.list_data)):
                    # print('clip: {}'.format(i))
                        list_of_img_of_same_record = [img[0] for img in self.list_of_rows_with_same_mode
                                              if img[1] == first_img[1]  # sessione
                                              and img[2] == first_img[2]  # gesture
                                              and img[3] == first_img[3]]  # record

                        center_of_list = math.floor(len(list_of_img_of_same_record) / 2)
                        crop_limit = math.floor(self.n_frames / 2)
                        start = center_of_list - crop_limit
                        end = center_of_list + crop_limit
                        list_of_img_of_same_record_cropped = list_of_img_of_same_record[
                                                             start: end + 1 if self.n_frames % 2 == 1 else end]
                        list_img = []
                        if self.mode == 'leap_motion_tracking_data':
                            list_of_json_frame = []
                            for js in list_of_img_of_same_record:
                                f_js = utilities.from_json_to_list(js)
                                if self.tracking_data_mod:
                                    f_js = utilities.extract_features_tracking_data(f_js)
                                list_img.append(np.asarray(f_js))
                        else:

                            for img_path in list_of_img_of_same_record_cropped:

                                if self.mode != 'depth_z':
                                    img = cv2.imread(img_path, 0 if not self.rgb else 1)
                                    img = cv2.resize(img, (self.resize_dim, self.resize_dim))
                                    if not self.rgb:
                                        img = np.expand_dims(img, axis=2)
                                elif self.mode == 'depth_z':
                                    f = gzip.GzipFile(img_path, "r")
                                    img = np.loadtxt(f)
                                    img = cv2.resize(img, (self.resize_dim, self.resize_dim))
                                    img = np.expand_dims(img, axis=2)
                                list_img.append(img)

                        if self.mode != 'leap_motion_tracking_data':
                            img_concat = np.concatenate(list_img, axis=2)
                            list_clip_to_norm.append(img_concat)
                        else:
                            list_clip_to_norm.append(np.vstack(list_img))
                    # print(list_clip_to_norm)
                    list_clip_to_norm = np.vstack(list_clip_to_norm)
                    self.mean = np.mean(list_clip_to_norm)
                    self.std = np.std(list_clip_to_norm)
                    np.savez("mean_std_{}.npz".format(self.mode), self.mean, self.std)
                    print('mean, std {} saved.'.format(self.mode))

            else: # load file
                npzfile = np.load("mean_std_{}.npz".format(self.mode))
                self.mean = npzfile['arr_0']
                self.std = npzfile['arr_1']
                print('mean, std {} loaded.'.format(self.mode))

        print('dataset_initialized')

    def __getitem__(self, index):
        """
               Args:
                   index (int): Index
               Returns:
                   tuple: (image, target) where target is class_index of the target class.
               """

        img_data = self.list_data[index]

        if self.mode == 'leap_motion_tracking_data': # in list data cè già il record aperto
            target = torch.LongTensor(np.asarray([int(img_data[1][5])]))
            img_data = img_data[0]

            if self.normalization_type:

                img_data = (np.asarray(img_data) - self.mean) / self.std

            return torch.Tensor(img_data), target

        target = torch.LongTensor(np.asarray([int(img_data[5])]))

        list_of_img_of_same_record = [img[0] for img in self.list_of_rows_with_same_mode
                                       if img[1] == img_data[1] # sessione
                                       and img[2] == img_data[2] # gesture
                                       and img[3] == img_data[3]]  # record

        list_of_img_of_same_record_cropped = []
        # if self.mode != 'leap_motion_tracking_data':
        center_of_list = math.floor(len(list_of_img_of_same_record) / 2)
        crop_limit = math.floor(self.n_frames / 2)
        start = center_of_list - crop_limit
        end = center_of_list + crop_limit
        list_of_img_of_same_record_cropped = list_of_img_of_same_record[
                                             start: end + 1 if self.n_frames % 2 == 1 else end]

        if self.model == 'C3D':
            if self.mode == 'depth_z':
                # clip = np.asarray([np.expand_dims(cv2.resize(np.loadtxt(gzip.GzipFile(frame, 'r')),
                #                                                      (112, 112)), axis=2)
                #                  for frame in list_of_img_of_same_record_cropped])
                clip = []
                for img_path in list_of_img_of_same_record_cropped:
                    f = gzip.GzipFile(img_path, "r")
                    img = np.loadtxt(f)
                    img = cv2.resize(img, (112, 112))
                    img = np.expand_dims(img, axis=2)
                    clip.append(img)
                clip = np.asarray(clip)
            elif self.mode == 'depth_ir':
                clip = np.asarray(
                    [np.expand_dims(cv2.resize(cv2.imread(frame, False), (112, 112)), axis=2) for frame in
                     list_of_img_of_same_record_cropped])

            elif self.mode == 'rgb':
                if self.rgb:
                    clip = np.asarray(
                        [cv2.resize(cv2.imread(frame, self.rgb), (112, 112)) for frame in
                         list_of_img_of_same_record_cropped]
                    )
                else: # gray scale
                    clip = np.asarray(
                        [np.expand_dims(cv2.resize(cv2.imread(frame, self.rgb), (112, 112)), axis=2) for frame in
                         list_of_img_of_same_record_cropped]
                    )

            clip = clip.transpose([3, 0, 1, 2])  # ch, fr, h, w
            clip = clip.astype(np.float32)
            if self.normalization_type is not None:
                clip = utilities.normalization(clip, self.mean, self.std, 1)

            return clip, target

        elif self.model == 'DeepConvLstm':
            # if self.mode == 'depth_z':
            #     clip = np.array([np.repeat(np.expand_dims(cv2.resize(np.loadtxt(gzip.GzipFile(frame, 'r')),
            #                                                          (self.resize_dim, self.resize_dim)), axis=2), 3, axis=2)
            #                      for frame in list_of_img_of_same_record_cropped])
            # else:
            #
            #
            #     clip = np.array([cv2.resize(cv2.imread(frame, self.rgb), (self.resize_dim, self.resize_dim)) for frame in
            #                      list_of_img_of_same_record_cropped])
            clip = utilities.create_clip(list_of_img_of_same_record=list_of_img_of_same_record, n_frames=self.n_frames,
                                         mode=self.mode, resize_dim=self.resize_dim, DeepConvLstm=True)

            if self.normalization_type is not None:
                clip = utilities.normalization(clip, self.mean, self.std, 1)

            # if not self.rgb:
            #     clip = np.expand_dims(clip, axis=-1)
            # clip = clip.transpose(0, 3, 1, 2)  # t, c, h, w  e poi b if batch_first
            clip = np.float32(clip)

            return clip, target

        else:
            list_img = []
            for img_path in list_of_img_of_same_record_cropped:
                if self.mode != 'depth_z':
                    img = cv2.imread(img_path, 0 if not self.rgb else 1)
                    # print("resize start:") #(480, 640, 3)
                    # print(img.shape)
                    img = cv2.resize(img, (self.resize_dim, self.resize_dim))
                    # #########
                    # ## Error #15: Initializing libiomp5md.dll
                    # os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
                    # print("print resize img")
                    # plt.figure()
                    # plt.subplot(1, 2, 1)
                    # plt.title("Original Image")
                    # plt.imshow(img[:, :, :1])
                    # img = cv2.resize(img, (self.resize_dim, self.resize_dim))
                    # # print("resize end:")  # (64, 64, 3)
                    # # print(img.shape)
                    # plt.subplot(1, 2, 2)
                    # plt.title("Transformed Image")
                    # plt.imshow(img[:, :, :1])
                    # plt.savefig("./resize_test.png")
                    # # exit("print resize picture.")
                    #########
                    if not self.rgb:
                        img = np.expand_dims(img, axis=2)
                elif self.mode == 'depth_z':
                    f = gzip.GzipFile(img_path, "r")
                    img = np.loadtxt(f)
                    img = cv2.resize(img, (self.resize_dim, self.resize_dim))
                    img = np.expand_dims(img, axis=2)
                list_img.append(img)

            img_concat = np.concatenate(list_img, axis=2)
            # print("1111")
            # print(img_concat.shape) #(64, 64, 120)
            if self.normalization_type is not None:
                img_concat = utilities.normalization(img_concat, self.mean, self.std, 1).astype(np.float32)
            # print(img_concat.shape) #(64, 64, 120)
            # print("2222")
            # img_concat = img_concat.transpose([2, 0, 1])
            if self.transforms is not None:
                ### no picture
                img_concat = self.transforms(img_concat)
                ### draw picture
                # ## Error #15: Initializing libiomp5md.dll
                # os.environ["KMP_DUPLICATE_LIB_OK"]="True"
                # print("print img")
                # plt.figure()
                # plt.subplot(1, 2, 1)
                # plt.title("Original Image")
                # plt.imshow(img_concat[:,:,:1])
                # # torch.Size([64, 64, 120]) --> torch.Size([120, 64, 64])
                # img_concat = self.transforms(img_concat)
                # print(img_concat.shape)
                # plt.subplot(1, 2, 2)
                # plt.title("Transformed Image")
                # print(img_concat.permute(1, 2, 0)[:,:,:1].shape)
                # plt.imshow(img_concat.permute(1, 2, 0)[:,:,:1])
                # plt.savefig("./transforms_test.png")
                # exit("print img end")
            if self.mode == 'depth_z':
                img_concat = img_concat.transpose([2, 0, 1])
                img_concat = img_concat.astype(np.float32)

            target = torch.LongTensor(np.asarray([int(img_data[5])]))

            return img_concat, target

    def __len__(self):
        return len(self.list_data)
