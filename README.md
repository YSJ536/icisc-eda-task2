# icisc-eda-task2
此文件是基于2023年集成电路EDA设计精英挑战赛-赛题二

imap为赛题官方平台，提供了几种优化算子优化组合逻辑  
imap平台：
> https://github.com/oscc-project/iMAP  

benchmark为训练集和测试集



## 环境依赖：
    tensorflow
    numpy

## 使用：
首先需要根据imap内readme激活python接口

之后使用命令

*python DQN.py input_file*  
input_file是数据集（aig文件）
## 算法说明

本算法使用了DQN强化学习优化组合逻辑，以优化算子为动作空间、电路特征为状态空间，有两个网络：主网络和目标网络，每迭代n次会将主网络的权重更新到目标网络

DQN.py是一个训练过程，训练模型会保存在iMAP-MAIN/output之中，在其中已经有了一个模型，是训练了10个数据集的一个模型
对于本算法，电路特征只提取了面积和延时，只有两个特征使得算法难以优化达到最佳