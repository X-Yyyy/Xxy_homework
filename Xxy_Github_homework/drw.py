import matplotlib.pyplot as plt
import numpy as np
# Vgg16_file = "./logs/gesture_recog_logs/txt_logs/Vgg16_rgb_prova_100.txt"
# LeNet_file = "./logs/gesture_recog_logs/txt_logs/LeNet_rgb_prova_100.txt"
# MultiRF2_file = "./logs/gesture_recog_logs/txt_logs/MultiRF2_Net_rgb_prova_100.txt"
# val_losses = []
# val_loss = []
# val_losses_avg6 = []  # 每6次的val_loss(avg)的平均值
# val_loss_6 = []  # 长度为6的列表，存储每6次的val_loss(avg)的值
# #count = 0  # 计数器实现每20个输出一次
#
# with open(MultiRF2_file, 'r') as f:
#     for line in f:
#         if "val_loss(avg):" in line:
#             #count += 1  # 计数器加1
#             #if count % 20 == 0:  # 每20个输出一次
#                 val_loss = float(line.split("val_loss(avg): ")[1].split()[0].strip('%'))  # strip函数将%去除
#                 #val_losses.append(val_loss)
#                 val_loss_6.append(val_loss)
#                 if len(val_loss_6) == 6:
#                     val_loss_avg6 = sum(val_loss_6) / 6
#                     val_losses_avg6.append(val_loss_avg6)
#                     val_loss_6 = val_loss_6[1:]  # 移除第一个元素
# plt.plot(np.arange(len(val_losses))*20, val_losses)# x轴乘以20是为了显示迭代次数
# plt.plot(np.arange(len(val_losses_avg6))*6, val_losses_avg6)
# plt.xlabel("Iteration")
# plt.ylabel("Validation Accuracy")
# plt.title("Validation Accuracy of LeNet Model")
# plt.show()
# # 保存图像
# plt.savefig('./logs/pic/val_acc_vgg16.png')
# # 显示图像
# plt.show()

import re
import matplotlib.pyplot as plt
import numpy as np

Vgg16_file = "./logs/gesture_recog_logs/txt_logs/Vgg16_rgb_prova_100.txt"
LeNet_file = "./logs/gesture_recog_logs/txt_logs/LeNet_rgb_prova_100.txt"
# LeNet_file = "./logs/gesture_recog_logs/txt_logs/LeNet_leap_motion_tracking_data_prova_100.txt"
# LeNet_file = "./logs/gesture_recog_logs/txt_logs/LeNet_tof_ir_prova.txt"
# MultiRF2_file = "./logs/gesture_recog_logs/txt_logs/MultiRF2_Net_rgb_prova_100.txt"
MultiRF2_file = "./logs/gesture_recog_logs/txt_logs/MultiRF2_Net_leap_motion_tracking_data_prova_100.txt"
# 读取vgg.txt文件中的数据
with open(MultiRF2_file, 'r') as f:
    lines = f.readlines()
# 从每行中提取需要的信息
epoch_data = {}
for line in lines:
    # if 'val_acc(avg)' in line:
    if 'val_loss(avg)' in line:
        epoch = int(re.findall(r'\[epoch: (\d+),', line)[0])
        # val_acc_avg = float(re.findall(r'val_acc\(avg\): (\d+\.\d+)', line)[0])
        val_loss = float(re.findall(r'val_loss\(avg\): (\d+\.\d+)', line)[0])
        if epoch in epoch_data:
            epoch_data[epoch].append(val_loss)
            # epoch_data[epoch].append(val_acc_avg)
        else:
            epoch_data[epoch] = [val_loss]
            # epoch_data[epoch] = [val_acc_avg]
# 计算每个epoch中val_acc(avg)的平均值
x = []
y = []
for epoch in epoch_data:
    # val_acc_avg_avg = np.mean(epoch_data[epoch]) # 计算平均值
    val_loss_avg_avg = np.mean(epoch_data[epoch])
    x.append(epoch)
    y.append(val_loss_avg_avg)
# for line in lines:
#     if 'loss(avg)' in line:
#         epoch = int(re.findall(r'\[epoch: (\d+),', line)[0])
#         loss = float(re.findall(r'loss\(avg\): (\d+\.\d+)', line)[0])
#         if epoch in epoch_data:
#             epoch_data[epoch].append(loss)
#         else:
#             epoch_data[epoch] = [loss]
# # 计算每个epoch平均值
# x = []
# y = []
# for epoch in epoch_data:
#     loss_avg_avg = np.mean(epoch_data[epoch])
#     x.append(epoch)
#     y.append(loss_avg_avg)
import matplotlib.pyplot as plt
fig = plt.figure()
plt.rc('font',family='Times New Roman')
ax = fig.add_subplot(111)
# 绘制图表
ax.plot(x, y)
import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.xlabel("Epoch")
# plt.ylabel("Validation Accuracy")
# plt.title("Validation Accuracy of MultiRF2 Model")
plt.ylabel("Validation Loss")
plt.title("Validation Loss of MultiRF2 Model")
# plt.title("Validation Loss of MultiRF2 Model")
# plt.ylabel("Loss")
# plt.title("Loss of LeNet Model")
# 保存图像
# plt.savefig('./logs/pic2/val_acc_MultiRF2__rgb.png')
# plt.savefig('./logs/pic/loss_LeNet_rgb.png')
plt.savefig('./logs/pic/loss_MultiRF2_leap_motion.png')
# plt.savefig('./logs/pic2/val_loss_LeNet__leap_motion_tracking_data.png')
# 显示图像
plt.show()