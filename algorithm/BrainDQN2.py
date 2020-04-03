import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from collections import deque
from time import time
from tensorflow.keras.utils import plot_model

# Hyper Parameters:
GAMMA = 0.99  # 奖励衰减值
EXPLORE = 30000.
FINAL_EPSILON = 0.01  # epsilon的最小值
INITIAL_EPSILON = 0.01  #  epsilon初始值
REPLAY_MEMORY = 25000  # 记忆库容量
BATCH_SIZE = 128  # 每次取样数
LEARNINGRATE = 5e-4


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        # initializer = tf.initializers.VarianceScaling(scale=2.0)
        initializer = tf.initializers.GlorotUniform()
        regularizer = tf.keras.regularizers.l2(l=0.01)
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.conv1_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.pool1_3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.conv1_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.conv1_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.pool1_6 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.fc1_8 = tf.keras.layers.Dense(500, activation='relu',
                                           kernel_initializer=initializer, kernel_regularizer=regularizer)

        self.conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.conv2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.pool2_3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.conv2_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.conv2_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.pool2_6 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.fc2_8 = tf.keras.layers.Dense(500, activation='relu',
                                           kernel_initializer=initializer, kernel_regularizer=regularizer)

        self.conv3_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 3), strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.conv3_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.pool3_3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.conv3_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.conv3_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.pool3_6 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.fc3_8 = tf.keras.layers.Dense(500, activation='relu',
                                           kernel_initializer=initializer, kernel_regularizer=regularizer)

        self.conv4_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.conv4_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.pool4_3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.conv4_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.conv4_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.pool4_6 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.fc4_8 = tf.keras.layers.Dense(500, activation='relu',
                                           kernel_initializer=initializer, kernel_regularizer=regularizer)

        self.flatten = tf.keras.layers.Flatten()
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.fc11 = tf.keras.layers.Dense(400, activation='relu',
                                          kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.fc12 = tf.keras.layers.Dense(2, kernel_initializer=initializer, kernel_regularizer=regularizer)

    def call(self, input_tensor, training=False):
        # print("\nis_trainning", training)
        x = input_tensor
        # print("x input:", x.shape)
        # mean, var = np.mean(x, axis=(0,1,2)), np.var(x, axis=(0, 1,2))
        # mean, var = tf.math.reduce_mean(x, axis=(0,1,2)), tf.math.reduce_variance(x, axis=(0,1,2))
        # print("batch_norm", mean, var)
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.pool1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.pool1_6(x1)
        x1 = self.flatten(x1)
        x1 = self.fc1_8(x1)

        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x2 = self.pool2_3(x2)
        x2 = self.conv2_4(x2)
        x2 = self.conv2_5(x2)
        x2 = self.pool2_6(x2)
        x2 = self.flatten(x2)
        x2 = self.fc2_8(x2)

        x3 = self.conv3_1(x)
        x3 = self.conv3_2(x3)
        x3 = self.pool3_3(x3)
        x3 = self.conv3_4(x3)
        x3 = self.conv3_5(x3)
        x3 = self.pool3_6(x3)
        x3 = self.flatten(x3)
        x3 = self.fc3_8(x3)

        x4 = self.conv4_1(x)
        x4 = self.conv4_2(x4)
        x4 = self.pool4_3(x4)
        x4 = self.conv4_4(x4)
        x4 = self.conv4_5(x4)
        x4 = self.pool4_6(x4)
        x4 = self.flatten(x4)
        x4 = self.fc4_8(x4)

        x5 = self.concat([x1, x2, x3, x4])
        x5 = self.fc11(x5)
        x5 = self.fc12(x5)
        return x5

class Model_Simple(tf.keras.Model):
    def __init__(self):
        super(Model_Simple, self).__init__()
        # initializer = tf.keras.initializers.VarianceScaling(scale=0.02) # 0.02 is best
        initializer = tf.initializers.GlorotUniform()
        regularizer = tf.keras.regularizers.l2(l=0.01)
        self.bn = tf.keras.layers.BatchNormalization(axis=3)
        self.conv1 = tf.keras.layers.Conv2D(filters=10, kernel_size=2, strides=1, padding='same',
                                              activation='relu', kernel_initializer=initializer)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.fc3 = tf.keras.layers.Dense(100, activation='relu',
                                           kernel_initializer=initializer)
        self.fc4 = tf.keras.layers.Dense(2, kernel_initializer=initializer)
        self.flatten = tf.keras.layers.Flatten()


    def call(self, input_tensor, training=False):
        # x = self.bn(input_tensor, training)
        x = input_tensor
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x =self.pool1(x)
        # print(x.shape)
        # x = self.conv2(x)
        # x = self.pool2(x)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = self.fc4(x)
        # print(x.shape)
        return x


class BrainDQN:
    def __init__(self, actions):
        # 初始化epsilon、行动个数
        self.epsilon = INITIAL_EPSILON
        self.initial_eps = INITIAL_EPSILON
        self.n_actions = actions
        self.train_step = 1
        self.cost_list = []
        self.replayMemory = deque()	#双向队列

        # init Q network
        self.eval_model = self.createQNetwork()
        self.target_model = self.createQNetwork()

        # saving and loading networks
        self.eval_model_path = "shen/saved_networks/eval_checkpoint"
        self.target_model_path = "shen/saved_networks/target_checkpoint"
        self.model_dir = os.path.dirname(self.eval_model_path)
        if os.path.exists("shen/saved_networks/checkpoint"):
            self.eval_model.load_weights(self.eval_model_path)
            self.target_model.load_weights(self.target_model_path)
            print("################Successfully loaded:", self.eval_model_path, self.target_model_path,
            "####################\n##############\n#################\n####################\n")
        else:
            print("################Could not find old network weights################\n################\n###########")

        # self.device = '/cpu:0'
        self.device = '/gpu:3'
        self.optimizer = tf.keras.optimizers.Adam(lr=LEARNINGRATE)
        self.eval_model.compile(optimizer=self.optimizer, loss='mse')
        self.eval_model._experimental_run_tf_function = False
        # self.target_model.compile(optimizer=self.optimizer, loss='mse')

    def createQNetwork(self):
        # model = Model_Simple()
        model = Model()
        return model

    def trainQNetwork(self):

        if self.train_step % 200 == 0:
            self.copyTargetQNetwork()
            print('copyNetWork-----------------------------------------------train_steps = ' + str(self.train_step))

        if self.train_step % 10000 == 0:
            self.eval_model.save_weights(self.eval_model_path)
            self.target_model.save_weights(self.target_model_path)


        with tf.device(self.device):
            minibatch = random.sample(self.replayMemory, BATCH_SIZE)
            state_batch = np.array([data[0] for data in minibatch])
            action_batch = np.array([data[1] for data in minibatch])
            reward_batch = np.array([data[2] for data in minibatch])
            next_state_batch = np.array([data[3] for data in minibatch])
            terminal_batch = np.array([data[4] for data in minibatch])
            # Target Q Network
            is_training = True
            # 直接call model 返回的是tensor 无法赋值， predict返回的numpy.ndarray
            q_eval = self.eval_model.predict(state_batch) # (batch_size, 2)
            q_next_batch = self.target_model.predict(next_state_batch)
            q_next4eval_batch = self.eval_model.predict(next_state_batch)
            q_actionmax_batch = np.argmax(q_next4eval_batch, axis=-1)

            q_target = q_eval.copy()
            action_index = np.argmax(action_batch, axis=-1)
            # print(action_batch.shape, q_next4eval_batch.shape, action_index.shape, q_actionmax_batch.shape)
            for i in range(BATCH_SIZE):
                terminal = terminal_batch[i]
                if terminal:
                    q_target[i, action_index[i]] = reward_batch[i]
                else:
                    q_target[i, action_index[i]] = reward_batch[i] + GAMMA * q_next_batch[i, q_actionmax_batch[i]]

            self.cost = self.eval_model.train_on_batch(state_batch, q_target)


        self.cost_list.append(self.cost)
        if self.train_step % 10000 == 0:
            # print("cost_list:", self.cost_list)
            plt.plot(self.cost_list)
            plt.xlabel("train_step")
            plt.ylabel("loss")
            plt.title("loss with batch_size 32")
            plt.savefig('shen/picture/' + "loss" + "_steps_" + str(self.train_step / 10000) + '_.png')
            # plt.savefig('shen/picture/' + "loss" + '_.png')
            plt.cla()
            self.cost_list=[]

        self.train_step += 1
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE  #


    def getAction(self, observation, day):
        action = np.zeros(self.n_actions)
        observation = observation[None,:,:,:]
        # if np.random.uniform() < self.epsilon: # 0.9 -> 0.001
        #     # 随机选择
        #     action_index = random.randrange(0, 10, 1)
        #     if action_index >= 8: # 先设成强制选择等待
        #         action_index = 1
        #     else:
        #         action_index = 0
        #     # action_index = 0

        QValue = self.eval_model.predict(observation)
        action_index = np.argmax(QValue)
        if day >= 86:
            action_index = 1
        action[action_index] = 1
        print('QValue:'+str(QValue) + " action:" + str(action_index) + " day:"+str(day))

        return action

    def store_transition(self, s, a, r, s_, terminal):
        self.replayMemory.append((s, a, r, s_, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()

    def copyTargetQNetwork(self):
        # for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
        #     target_layer.set_weights(eval_layer.get_weights())
        self.target_model.set_weights(self.eval_model.get_weights())

if __name__ == "__main__":
    with(tf.device(":\cpu0")):
        model = Model()
        model.load_weights("../shen/saved_networks/eval_checkpoint")
        plot_model(model, to_file='model.png')
