~Xiao xingyi~
软件研23-2班
学号：472321759
github作业!!!

配置环境，pytorch
下载车载手势识别的数据集，并存放在本地路径。
    下载链接如下：
    http://imagelab.ing.unimore.it/briareo
1.编写读取数据集的Python代码：
    set_up_dataset.py：该代码由于生成数据文件。
    是根据dataset_type和dataset_path来读取数据集所在的路径，
    然后保存路径信息到.txt文件中， 生成以dataset_path为标识的数据文件。
    数据文件共保存7个字段，有：img_path,session_id,gesture_id,record,mode,label,first
    最终生成三个数据文件：训练集、验证集以及测试集。
2.模型训练主函数：
    main.py： 该代码将完成以下几个子任务
    ** 配置超参数，比如常见的参数有：模型结构，批大小，训练周期数，优化器，学习率，是否继续训练，数据集，输入大小，分类类别数等等
    ** 判断Python环境是否支持GPU版本的Pytorch，不支持则采用CPU
    ** 设置随机种子
    ** 加载训练和验证集数据
    ** 初始化模型，并计算模型参数大小
    ** 设置优化器
    ** 定义损失函数：交叉熵函数
    ** 设置检查点：加载已训练模型的字典、优化器、已训练次数、损失值以及准确率等数据
    ** 设置log函数，记录实验日志信息
    ** 设置训练器，进行模型的训练和验证
    ** 计算模型准确率以及训练时间，并写入log文件中
3.模型推理主函数代码：
    eval.py：
    ** 配置超参数，
    ** 判断Python环境是否支持GPU版本的Pytorch，不支持则采用CPU
    ** 设置随机种子
    ** 加载测试集数据
    ** 加载最优模型进行模型测试
    ** 保存推理结果并写入文件
