import torch
from torch import nn, optim
import argparse

from LeNet_improve import MultiRF2_Net, MultiRF2_Net_Leap
from dataloader import GesturesDataset
from models import LeNet, Vgg16, C3D, LeNet_Leap
from trainer import Trainer
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter #pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
from torchvision import models
import time, os,sys
import random
import numpy as np
# from ConvGru import ConvGRU
sys.path.append("./")
"""
--model LeNet
--mode rgb(it is rgb dataset)/leap_motion_tracking_data(it is leap motion dataset)/depth_ir(it is tof dataset)
"""
# 导入模块的搜索路径，包含已经添加到系统的环境变量路径
parser = argparse.ArgumentParser(description='PyTorch conv2d')
# 添加程序参数信息
parser.add_argument('--model', type=str, default='MultiRF2_Net', ### LeNet MultiRF2_Net ResNet34P
                    help='model of NN')
parser.add_argument('--final_layer', type=str, default='fc',
                    help='final layer of rnn')
parser.add_argument('--pretrained', default=True,
                    help='pretrained net')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--opt', type=str, default='SGD',
                    help="Optimizer (default: SGD)")
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--dn_lr', action='store_true', default=False,
                    help="adjust dinamically lr")
# 让网络的初始权值处在一个合适的状态，momentum动量越大时，转换为势能的能量也就越大，越有可能摆脱局部凹域的束缚，进入全局凹域。主要用在权重更新时
# SGD随机梯度下降，一次迭代只需要用一个样本计算梯度。BGD每次迭代都用所有样本，所以速度在大数据量很慢
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
# 权重衰减，Adam泛化性不如SGD with Momentum
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='M',
                    help='Adam weight_decay (default: 0.0001')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1994, metavar='S',
                    help='random seed (default: 1994)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before checkpointing')
parser.add_argument('--resume', action='store_true', default=True,
                    help='resume training from checkpoint')
parser.add_argument('--n_workers', type=int, default=0,
                    help="number of workers")
parser.add_argument('--mode', type=str, default='rgb',## rgb(64x64)/L_raw(64x64)/L_undistorted(64x64)/R_raw(64x64)/R_undistorted(64x64)/tof_ir(64x64)  ###leap_motion_tracking_data(json) tof_depth(npz)
                    help='mode of dataset')
parser.add_argument('--gray_scale', action='store_true', default=False)
parser.add_argument('--n_frames', type=int, default=40,
                    help='number of frames per input')
parser.add_argument('--input_size', type=int, default=64, # 227 alexnet, 64 lenet, 224 vgg16 and Resnet, denseNet
                    help='input size')
parser.add_argument('--train_transforms', action='store_true', default=False,
                    help="training transforms")
parser.add_argument('--n_classes', type=int, default=12,
                    help='number of frames per input')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='hidden size of rnn')
parser.add_argument('--n_layers', type=int, default=4,
                    help='n layers of rnn')
parser.add_argument('--tracking_data_mod', action='store_true', default=False)

# parser.add_argument('--weight_dir', type=str)
# parser.add_argument('--exp_name', type=str, default="prova")
parser.add_argument('--exp_name', type=str, default="test")
# ArgumentParser通过parse_args()方法，解析参数
args = parser.parse_args()

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6
def main():

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    rgb = False
    if args.mode == 'rgb':
        rgb = True

    if args.gray_scale:
        rgb = False

    if args.tracking_data_mod is True:
        args.input_size = 192

    # DATALOADER
    # dataset_type = "rgb" # "rgb" "tof"
    # dataset_type = "leap_motion" # "rgb" "tof"
    dataset_type= ""
    if args.mode == "rgb":
        dataset_type = "rgb_mini" #"rgb"
    elif args.mode == "L_raw" or args.mode == "L_undistorted" or args.mode == "R_raw" or args.mode == "R_undistorted" or args.mode == "leap_motion_tracking_data":
        dataset_type = "leap_motion"
    elif args.mode == "tof_ir":
        dataset_type = "tof"
    else:
        print("args.mode error!")
        exit()
    print("模型为{}，数据集mode为{}，rgb为{}".format(args.model,args.mode,rgb))
    # 构造dataset对象
    train_dataset = GesturesDataset(model=args.model, csv_path='./csv_train_dataset_'+str(dataset_type)+'.txt', train=True, mode=args.mode, rgb=rgb,
                                    normalization_type=1,
                                    n_frames=args.n_frames, resize_dim=args.input_size,
                                    transform_train=args.train_transforms, tracking_data_mod=args.tracking_data_mod)
    print("train_dataset 中共有 " + str(len(train_dataset)) + " 组数据")
    # 通过dataloader构造迭代对象
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

    validation_dataset = GesturesDataset(model=args.model, csv_path='./csv_validation_dataset_'+str(dataset_type)+'.txt', train=False, mode=args.mode, rgb=rgb, normalization_type=1,
                                   n_frames=args.n_frames, resize_dim=args.input_size, tracking_data_mod=args.tracking_data_mod)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    print("validation_dataset 中共有 " + str(len(validation_dataset))+ " 组数据")
    if len(train_dataset)==0 or len(validation_dataset)==0:
        exit("Dataset is Null !!!")

    in_channels = args.n_frames if not rgb else args.n_frames * 3
    n_classes = args.n_classes

    if args.model == 'LeNet':
        if args.mode == "leap_motion_tracking_data":
            in_channels = 1
            model = LeNet_Leap(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)
        else:
            model = LeNet(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)
    elif args.model == "Vgg16":
        model = Vgg16(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)

    elif args.model == "ResNet34P":
        model = models.resnet34(pretrained=args.pretrained)
        for params in model.parameters():
            params.requires_grad = False
        model._modules['conv1'] = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        model = model.to(device)
    # C3D
    elif args.model == 'C3D':
        if args.pretrained:
            model = C3D(rgb=rgb, num_classes=args.n_classes)
            print('ok')

            # model.load_state_dict(torch.load('c3d_weights/c3d.pickle', map_location=device), strict=False)
            # # for params in model.parameters():
            #     # params.requires_grad = False

            model.conv1 = nn.Conv3d(1 if not rgb else 3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            model.fc6 = nn.Linear(16384, 4096)  # num classes 28672 (112*200)
            model.fc7 = nn.Linear(4096, 4096)  # num classes
            model.fc8 = nn.Linear(4096, n_classes)  # num classes

            model = model.to(device)

    elif args.model == 'MultiRF2_Net':
        if args.mode == "leap_motion_tracking_data":
            in_channels = 1
            model = MultiRF2_Net_Leap(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)
        else:
            model = MultiRF2_Net(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)
    # elif args.model == 'LeNet_Improve':
    #     model = LeNet_Improve(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)
    #     # print(model)
    else:
        raise NotImplementedError
    print(model)
    print('Loading model parameters: {:.3f} Mb'.format(check_parameters(model)))
    if args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum)

    elif args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_function = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load("./weights/gesture_recog_weights/checkpoint_{}.pth.tar".format(args.model))
        # checkpoint = torch.load("./weights/gesture_recog_weights/{}_{}_prova/best_val_loss_checkpoint_{}_{}_prova.pth.tar".format(args.model,args.mode,args.model,args.mode))
        # checkpoint = torch.load("./weights/gesture_recog_weights/{}_{}_prova/best_val_acc_checkpoint_{}_{}_prova.pth.tar".format(args.model,args.mode,args.model,args.mode))
        # 加载模型参数，pytorch把所有的模型参数用一个内部定义的dict进行保存，自称为“state_dict”，就是不带模型结构的模型参数
        model.load_state_dict(checkpoint['state_dict'])
        # 保存优化器的 state_dict 也很重要, 因为它包含作为模型训练更新的缓冲区和参数
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        resume_loss = checkpoint['running_loss']
        resume_acc = checkpoint['accuracy']

        # print("Resuming state:\n-epoch: {}\n{}".format(start_epoch, model))
        print("Resuming state:\n-epoch: {}".format(start_epoch))

    #name experiment
    # format格式化函数:指定输出的格式和内容，举个例子，"{} {}".format("hello", "world")不设置指定位置，按默认顺序输出'hello world'
    personal_name = "{}_{}_{}".format(args.model, args.mode, args.exp_name)
    info_experiment = "{}".format(personal_name)
    log_dir = "./logs/gesture_recog_logs/exps"
    weight_dir = personal_name
    if not os.path.exists("./logs/gesture_recog_logs/txt_logs"):
        os.makedirs("./logs/gesture_recog_logs/txt_logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # w覆盖
    # log_file = open("{}/{}.txt".format("./logs/gesture_recog_logs/txt_logs", personal_name), 'w+')
    log_file = open("{}/{}.txt".format("./logs/gesture_recog_logs/txt_logs", personal_name), 'a+')
    log_file.write(personal_name + "\n\n")
    if personal_name:
        exp_name = (("exp_{}_{}".format(time.strftime("%c"), personal_name)).replace(" ", "_")).replace(":", "-")
    else:
        exp_name = (("exp_{}".format(time.strftime("%c"), personal_name)).replace(" ", "_")).replace(":", "-")
    writer = SummaryWriter("{}".format(os.path.join(log_dir, exp_name)))

    # add info experiment
    writer.add_text('Info experiment',
                    "model:{}"
                    "\n\npretrained:{}"
                    "\n\nbatch_size:{}"
                    "\n\nepochs:{}"
                    "\n\noptimizer:{}"
                    "\n\nlr:{}"
                    "\n\ndn_lr:{}"
                    "\n\nmomentum:{}"
                    "\n\nweight_decay:{}"
                    "\n\nn_frames:{}"
                    "\n\ninput_size:{}"
                    "\n\nhidden_size:{}"
                    "\n\ntracking_data_mode:{}"
                    "\n\nn_classes:{}"
                    "\n\nmode:{}"
                    "\n\nn_workers:{}"
                    "\n\nseed:{}"
                    "\n\ninfo:{}"
                    "".format(args.model, args.pretrained, args.batch_size, args.epochs, args.opt, args.lr, args.dn_lr, args.momentum,
                              args.weight_decay, args.n_frames, args.input_size, args.hidden_size, args.tracking_data_mod,
                              args.n_classes, args.mode, args.n_workers, args.seed, info_experiment))
    if args.resume:
        trainer = Trainer(model=model, loss_function=loss_function, optimizer=optimizer, train_loader=train_loader,
                          validation_loader=validation_loader,
                          batch_size=args.batch_size, initial_lr=args.lr, device=device, writer=writer,
                          personal_name=personal_name, log_file=log_file,
                          weight_dir=weight_dir, dynamic_lr=args.dn_lr ,resume_loss_state=resume_loss,resume_acc_state=resume_acc,resume_flag = args.resume)
    else:
        trainer = Trainer(model=model, loss_function=loss_function, optimizer=optimizer, train_loader=train_loader,
                          validation_loader=validation_loader,
                          batch_size=args.batch_size, initial_lr=args.lr, device=device, writer=writer,
                          personal_name=personal_name, log_file=log_file,
                          weight_dir=weight_dir, dynamic_lr=args.dn_lr)


    print("experiment: {}".format(personal_name))
    start = time.time()
    for ep in range(start_epoch, args.epochs):
        trainer.train(ep)
        trainer.val(ep)


    classes = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11']
    for i in range(args.n_classes):
        print('Accuracy of {} : {:.3f}%%'.format(
            classes[i], 100 * trainer.class_correct[i] / trainer.class_total[i]))

    end = time.time()
    h, rem = divmod(end - start, 3600)
    m, s, = divmod(rem, 60)
    print("\nelapsed time (ep.{}):{:0>2}:{:0>2}:{:05.2f}".format(args.epochs, int(h), int(m), s))

# 输出准确率
    log_file.write("\n\n")
    for i in range(args.n_classes):
        log_file.write('Accuracy of {} : {:.3f}%\n'.format(
            classes[i], 100 * trainer.class_correct[i] / trainer.class_total[i]))
    log_file.close()


if __name__ == '__main__':
    main()
