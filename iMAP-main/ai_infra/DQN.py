import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from collections import deque
import sys
import os
import time
import re
from imap_engine import EngineIMAP

class Demo(object):

    def __init__(self, input_file) -> None:
        self.input_file = input_file
        self.engine = EngineIMAP(input_file, input_file+'.seq')

    def _opt_rewrite(self):
        self.engine.rewrite()
        self.engine.add_sequence('rewrite')

    def _opt_refactor(self):
        self.engine.refactor(zero_gain=True)
        self.engine.add_sequence('refactor -z')

    def _opt_depth(self):
        self.engine.balance()
        self.engine.add_sequence('balance')

    def _opt_lut(self):
        self.engine.lut_opt()
        self.engine.add_sequence('lut_opt')
    def read_aig(self):
        self.engine.read()



class HistoryDemo(Demo):
    '''
    A demo with history and choice mapping.
    '''
    def __init__(self, input_file, output_file) -> None:
        super().__init__(input_file, output_file)

    def _history_empty(self):
        return self.engine.history(size=True) == 0

    def _history_full(self):
        return self.engine.history(size=True) == 5

    def _history_add(self):
        self.engine.history(add=True)
        self.engine.add_sequence('history -a')

    def _history_replace(self, idx):
        self.engine.history(replace=idx)
        self.engine.add_sequence(f'history -r {idx}')
    
    
# 假设 N 是从aig文件中提取的特征数量
#class DQNTRAINer
demo = Demo(sys.argv[1])
demo.read_aig()
N = 2 # 例如，有2个特征
num_actions = 4
# 创建DQN模型
# 主网络
main_model = Sequential([
    Dense(64, activation='relu', input_shape=(N,)),  
    Dense(64, activation='relu'),
    Dense(num_actions)  # 动作的数量
])
main_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 目标网络
target_model = Sequential([
    Dense(64, activation='relu', input_shape=(N,)),  
    Dense(64, activation='relu'),
    Dense(num_actions)
])
target_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model = load_model('../output/my_model.h5')

# 初始时，将主网络的权重复制到目标网络
main_model.set_weights(model.get_weights())
target_model.set_weights(model.get_weights())


# 经验回放缓冲区
experience_replay_buffer = deque(maxlen=2000)

# 探索率
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.95

# 学习过程参数
gamma = 0.95  # 折扣因子
batch_size = 32
#reward function
def calculate_reward(old_value, new_value):
    if new_value < old_value:
        # value减少，给予正奖励
        reward = 10 * (old_value - new_value) / old_value
    else:
        # value增加或不变，给予负奖励或较小的正奖励
        reward = -5 * (new_value - old_value) / old_value
    return reward
demo.engine.print_stats()
with open('read.txt','r') as file:
        first_line = file.readline()
        numbers = first_line.split()
        numbers = [int(num) for num in numbers]
old_value = 0.4 * numbers[2] + 0.6 * numbers[3]
# 定义一个函数来模拟环境的反馈
def step(action, old_value):
    # 执行动作，返回新状态的特征和奖励
    if action == 0:
        # 执行 balance 动作
        demo._opt_depth()
    elif action == 1:
        # 执行 refactor 动作
        demo._opt_refactor()
    elif action == 2:
        # 执行 rewrite 动作
        demo._opt_rewrite()
    elif action == 3:
        # 执行 lutopt 动作
        demo._opt_lut()
    # 这里需要您自己定义如何根据动作更新aig文件，并提取新的特征
    demo.engine.print_stats()
    with open('read.txt','r') as file:
        first_line = file.readline()
        numbers = first_line.split()
        numbers = [int(num) for num in numbers]
    new_value = 0.4 * numbers[2] + 0.6 * numbers[3]
    reward = calculate_reward(old_value,new_value)  # 奖励
     # 是否结束
    if old_value <= new_value:
        done = True 
    else:
        done = False
    old_value = new_value  #update value
    new_state_features = numbers[-2:]
    return new_state_features, reward, done, old_value

# 训练过程
for episode in range(100):  # 假设有100个训练回合
    with open('read.txt','r') as file:
        first_line = file.readline()
        numbers = first_line.split()
        numbers = [int(num) for num in numbers]
    state = (numbers[2],numbers[3])  # 初始化状态
    total_reward = 0

    while True:  # 或者设置一些终止条件
        if np.random.rand() <= epsilon:
            action = np.random.randint(4)  # 探索新动作
        else:
            action = np.argmax(main_model.predict(np.array([state]))[0])  # 利用已学知识

        next_state, reward, done, old_value = step(action,old_value)  # 执行动作并观察结果
        total_reward += reward

        # 存储经验
        experience_replay_buffer.append((state, action, reward, next_state, done))

        state = next_state  # 更新状态
        next_state_reshaped = np.array(next_state).reshape(-1)
        # 经验回放
        if len(experience_replay_buffer) > batch_size:
            minibatch = random.sample(experience_replay_buffer, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target += gamma * np.amax(target_model.predict(np.array([next_state_reshaped]))[0])
                target_f = main_model.predict(np.array([state]))
                target_f[0][action] = target
                main_model.fit(np.array([state]), target_f, epochs=1, verbose=0)


            if done:
                break
    demo.engine.map_fpga()
    demo.engine.print_stats(type=1)
    # 更新探索率
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    target_update_frequency = 10    #update one time every 10 epoch 

    if episode % target_update_frequency == 0:
        target_model.set_weights(main_model.get_weights())

    print(f"Episode: {episode}, Total Reward: {total_reward}")

# 模型训练完成
main_model.save('../output/my_model.h5')  # 保存模型到一个 HDF5 文件
