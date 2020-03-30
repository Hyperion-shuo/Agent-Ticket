import numpy as np
import tensorflow.compat.v1 as tf
import random
from collections import deque
import matplotlib.pyplot as plt

# Hyper Parameters:
GAMMA = 0.99 # 奖励衰减值
EXPLORE = 10000.
EXPLORE1 = 5000. # 逐步减小epsilon
EXPLORE2 = 5000. # 逐步减小epsilon
FINAL_EPSILON = 0.01 # epsilon的最小值
INITIAL_EPSILON = 0.9 #  epsilon初始值
REPLAY_MEMORY = 10000 # 记忆库容量
BATCH_SIZE = 64 # 每次取样数
THRESHOLD = 0.0
LEARNINGRATE = 1e-3

class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity

        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype = tuple)


    def add(self, p, data):
        # 多了挤掉前面的
        tree_idx = self.data_pointer + self.capacity - 1
        # print("datatype" + str(type(data)))
        # for i in range(5):
            # print("datashape" + str(i) + " " + str(data[i].shape))
        self.data[self.data_pointer] = data
        # print(self.data[self.data_pointer])
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx!= 0:
            tree_idx = (tree_idx - 1) // 2 # 商取整
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        '''
            0
           1  2
          3 4 5 6
        '''
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        # print("leaf_sample_data", self.data[data_idx])
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]

class Memory(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):                                             #  np.empty((n, np.array(self.tree.data[0]).size))
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,), dtype=tuple), np.empty((n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # 最大为1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p # 0?
        for i in range(n):
            a, b = i * pri_seg, (i + 1) * pri_seg
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta) # 这步不太明白
            #         b_memory[i, :] 如果用 二维ndarray 存
            b_idx[i], b_memory[i] = idx, data # memory tuple 数组， 第一维batch__size, memory里四元组
            # print("b_memory" + str(i), b_memory[i])
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_index, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper) # 作用不明白
        ps = np.power(clipped_errors, self.alpha)
        # print("tree_index type", type(tree_index),"tree_index shape", tree_index.shape)
        # print("ps type", type(ps), "ps shape", ps.shape)
        for ti, p in zip(tree_index, ps):
            self.tree.update(ti, p)


class BrainDQN:
    def __init__(self,
                 actions,
                 prioritized=False):
        # 初始化epsilon、行动个数
        self.epsilon = INITIAL_EPSILON
        self.INITIAL_EPSILON = INITIAL_EPSILON
        self.n_actions = actions
        self.prioritized = prioritized
        self.train_step = 1

        #用cost来画图
        self.cost_list = []

        # 解决tf2.0没有计算图的问题
        tf.disable_eager_execution()

        # 初始化记忆库
        if self.prioritized:
            self.replayMemory = Memory(capacity=REPLAY_MEMORY)
        else:
            self.replayMemory = deque()	#双向队列

        # init Q network
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1,\
        self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T,\
        self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()
        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        # checkpoint = tf.train.get_checkpoint_state("shen/saved_networks")
        # if checkpoint and checkpoint.model_checkpoint_path:
        #     self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
        # else:
        print("Could not find old network weights")

    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8, 2, 1, 10])
        b_conv1 = self.bias_variable([10])

        W_fc1 = self.weight_variable([800, 100])
        b_fc1 = self.bias_variable([100])

        W_fc2 = self.weight_variable([100, self.n_actions]) # [200, 2]
        b_fc2 = self.bias_variable([self.n_actions]) # [2]

        # input layer
        stateInput = tf.placeholder("float", [None, 87, 2, 1])
        # 88 x 2 x 1

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 1) + b_conv1)
        # 80 x 1 x 10

        # h_pool1 = self.max_pool_4x4(h_conv1)
        # 20 x 1 x 10

        h_conv2_flat = tf.reshape(h_conv1, [-1, 800])
        # 200 x 1

        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        # 100 x 2
        # Q Value layer
        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2
        # 2 x 1

        return stateInput, QValue, W_conv1, b_conv1, W_fc1, b_fc1, W_fc2, b_fc2

    def createQNetwork2(self):
        # network weights
        W_conv1 = self.weight_variable([8, 2, 1, 32])
        b_conv1 = self.bias_variable([32])

        W_fc1 = self.weight_variable([800, 100])
        b_fc1 = self.bias_variable([100])

        W_fc2 = self.weight_variable([100, self.n_actions]) # [200, 2]
        b_fc2 = self.bias_variable([self.n_actions]) # [2]

        # input layer
        stateInput = tf.placeholder("float", [None, 87, 2, 1])
        # 88 x 2 x 1

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 1) + b_conv1)
        # 80 x 1 x 10

        # h_pool1 = self.max_pool_4x4(h_conv1)
        # 20 x 1 x 10

        h_conv2_flat = tf.reshape(h_conv1, [-1, 800])
        # 200 x 1

        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        # 100 x 2
        # Q Value layer
        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2
        # 2 x 1

        return stateInput, QValue, W_conv1, b_conv1, W_fc1, b_fc1, W_fc2, b_fc2

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.n_actions])
        if self.prioritized:
            self.yInput = tf.placeholder("float", [None, 1])
            # self.ISWeights = tf.placeholder(tf.float32, [None, 1], name="ISWeights")
            self.ISWeights = tf.placeholder("float", [None, 1])
            Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)  # 求和
            self.abs_errors = tf.reduce_sum(tf.abs(self.yInput - Q_Action), axis=1)
            self.cost = tf.reduce_mean(self.ISWeights * tf.square(self.yInput - Q_Action))  # 求均值
        else:
            self.yInput = tf.placeholder("float", [None])
            Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1) #求和
            self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action)) # 求均值

        # 输出loss用，注释掉后trainQNetwork也的feeddict也要注释
        # 单独的这段代码，不再后面的计算图中调用c_print无法计算cost
        # c_print = tf.Print(self.cost, ["cost:", self.cost, "################################"])
        # self.cost = self.cost + c_print - c_print

        self.trainStep = tf.train.AdamOptimizer(LEARNINGRATE).minimize(self.cost)

    def trainQNetwork(self):

        if self.train_step % 500 == 0:
            self.copyTargetQNetwork()
            print('copyNetWork--------------------------------------------------------train_steps = ' + str(
                self.train_step))

        if self.train_step % 10000 == 0:
            self.saver.save(self.session, 'shen/saved_networks/' + 'network' + '-dqn', global_step=self.train_step)

        # Step 1: 从记忆库随机取数据
        if self.prioritized:
            tree_idx, minibatch, ISWeights = self.replayMemory.sample(BATCH_SIZE)
            # print("minbatch_content",minibatch)
            # print("minibatch type" + str(type(minibatch)))
            # print("minibatch shape"+ str(minibatch.shape))
            state_batch = np.empty((BATCH_SIZE, 88, 2, 1))
            action_batch = np.empty((BATCH_SIZE, 2))
            reward_batch = np.empty((BATCH_SIZE, 1))
            nextState_batch = np.empty((BATCH_SIZE, 88, 2, 1))
            for i in range(BATCH_SIZE):
                state_batch[i] = minibatch[i][0]
                action_batch[i] = minibatch[i][1]
                reward_batch[i] = minibatch[i][2]
                nextState_batch[i] = minibatch[i][3]
            # print("state_batch type" + str(type(state_batch)))
            # print("state_batch shape" + str(state_batch.shape))
            # print(state_batch)
        else:
            minibatch = random.sample(self.replayMemory, BATCH_SIZE)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate（计算） y
        y_batch = []
        # Target Q Network
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch})
        QValueEval_batch = self.QValue.eval(feed_dict={self.stateInput: nextState_batch})
        QActionMax_batch = np.argmax(QValueEval_batch, axis=1) # 64 * 2 变 64 * 1 取最大值的下标
        # print("QActionMax shape", QActionMax_batch.shape)
        # print("QValue_batch shape", QValue_batch.shape)
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:                               # GAMMA * QValue_batch[i][QActionMax[i]
                # y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))
                y_batch.append(reward_batch[i] + GAMMA * QValue_batch[i][QActionMax_batch[i]])

        if self.prioritized:
            y_batch = np.mat(y_batch)
            y_batch = y_batch.reshape((BATCH_SIZE, 1))
            # print("ISWeith shape", ISWeights.shape)
            # print("y_batch shape", y_batch.shape)
            _, abs_errors, cost = self.session.run([self.trainStep, self.abs_errors, self.cost], feed_dict={
                self.yInput: y_batch,
                self.actionInput: action_batch,
                self.stateInput: state_batch,
                self.ISWeights: ISWeights
            })

            self.cost_list.append(cost)
            # print("cost_list:", self.cost_list)
            if self.train_step % 10000 == 0:
                # print("cost_list:", self.cost_list)
                plt.plot(self.cost_list)
                plt.ylim(1000)
                plt.xlabel("train_step")
                plt.ylabel("loss")
                plt.title("loss with batch_size 32")
                plt.savefig('shen/picture/' + "loss" + "_steps_" + str(self.train_step / 10000) + '_.png')
                plt.cla()

            # print("abs_errors", abs_errors)
            self.replayMemory.batch_update(tree_idx, abs_errors)
        else:
            '''
            self.trainStep.run(feed_dict={
                self.yInput: y_batch,
                self.actionInput: action_batch,
                self.stateInput: state_batch
            })
            '''
            _, cost = self.session.run([self.trainStep, self.cost], feed_dict={
                self.yInput: y_batch,
                self.actionInput: action_batch,
                self.stateInput: state_batch
            })

            # 打印loss
            # createtrainmethod 里相应的cost计算图也要同时定义
            # 注释掉时createtrainmethod 里也要注释
            '''
            self.session.run(tf.Print(self.cost, ["cost:", self.cost]), feed_dict={
                    self.yInput: y_batch,
                    self.actionInput: action_batch,
                    self.stateInput: state_batch
                })
            cost = self.session.run([self.cost],feed_dict={
                    self.yInput: y_batch,
                    self.actionInput: action_batch,
                    self.stateInput: state_batch
                })
            '''

            self.cost_list.append(cost)
            # print("cost_list:", self.cost_list)
            if self.train_step % 10000 == 0:
                # print("cost_list:", self.cost_list)
                plt.plot(self.cost_list)
                plt.ylim(ymax=5000)
                plt.xlabel("train_step")
                plt.ylabel("loss")
                plt.title("loss with batch_size 32")
                # plt.savefig('shen/picture/' + "loss" + "_steps_" + str(self.train_step / 10000) + '_.png')
                plt.savefig('shen/picture/' + "loss" + '_.png')
                plt.cla()

        self.train_step += 1

        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE  #

        '''
        if self.epsilon > THRESHOLD:
            self.epsilon -= (self.INITIAL_EPSILON - THRESHOLD) / EXPLORE1  # 开始先强制多探索
        else:
            self.epsilon -= (self.INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE2 # 再正常衰减
        '''

    def getAction(self, observation, day):
        # 统一 observation 的 shape (1, size_of_observation)
        # 解决Cannot feed value of shape (88, 2, 1) for Tensor 'Placeholder:0', which has shape '(?, 88, 2, 1)'
        observation = observation[np.newaxis, :] # 本来第一维为样本数， 单次估计则为1 ，要加上那一维
        action = np.zeros(self.n_actions)

        # if np.random.uniform() < self.epsilon: # 0.9 -> 0.001
        #     # 随机选择
        #     action_index = random.randrange(0, 10, 1)
        #     if action_index >= 8: # 先设成强制选择等待
        #         action_index = 1
        #     else:
        #         action_index = 0
        #     # action_index = 0
        # else:
            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
        QValue = self.QValue.eval(feed_dict={self.stateInput: observation})[0] # 等价于 session.run(QValue)
        action_index = np.argmax(QValue)
        if day >= 86:
            action_index = 1

        # 选择网络选的动作
        action[action_index] = 1

        # 输出打印出来  同时存到txt中
        # QValue = self.QValue.eval(feed_dict={self.stateInput: observation})[0]
        print('QValue:'+str(QValue) + " action:" + str(action_index) + " day:"+str(day))
        # self.file.write('QValue:'+str(QValue) + " action:" + str(action_index) + " day:"+str(day) + " epsilon:%.3f" % self.epsilon + "\n")

        return action

    def store_transition(self, s, a, r, s_, terminal):
        if self.prioritized:
            transition = (s, a, r, s_, terminal)
            self.replayMemory.store(transition)
        else:
            self.replayMemory.append((s, a, r, s_, terminal))
            if len(self.replayMemory) > REPLAY_MEMORY:
                self.replayMemory.popleft()

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    # 供构造network调用部分
    def weight_variable(self, shape):
        # 从截断的正态分布中输出随机值
        initial = tf.truncated_normal(shape, stddev=0.01)
        # 在TensorFlow的世界里，变量的定义和初始化是分开的，
        # 所有关于图变量的赋值和计算都要通过tf.Session的run来进行。
        # 想要将所有图变量进行集体初始化时应该使用tf.global_variables_initializer。
        # tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

    def max_pool_4x4(self, x):
        # 维度增加一维
        # n x 80 x 2 x 10
        # ksize：池化窗口的大小   strides：窗口在每一个维度上滑动的步长
        ret = tf.nn.max_pool(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding="VALID")
        # n x 20 x 1 x 10
        return ret