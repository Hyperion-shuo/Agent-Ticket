import numpy as np
import tensorflow as tf
import copy
np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_actions=2,
            n_features=87,
            learning_rate=0.01,
            reward_decay=0.95,
            prob_clip=0.06,
            output_graph=False,
    ):
        #动作空间的维数
        self.n_actions = n_actions
        #状态特征的维数
        self.n_features = n_features
        #学习速率
        self.lr = learning_rate
        #回报衰减率
        self.gamma = reward_decay
        #一条轨迹的观测值，动作值，和回报值
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]
        self.ep_length = []
        #创建策略网络
        self._build_net()

        self.prob_clip = prob_clip
        #启动一个默认的会话
        self.sess = tf.Session()
        # if output_graph:
        #     tf.summary.FileWriter("logs/", self.sess.graph)
        # 初始化会话中的变量
        self.sess.run(tf.global_variables_initializer())
    #创建策略网络的实现
    def _build_net(self):
        with tf.name_scope('input'):
            #创建占位符作为输入
            self.tf_obs = tf.placeholder(tf.float32, [None,self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None,], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
            # 构建一个向量，这个向量专门用来存储每一个样本中的timesteps的数目，这个是核心所在
            self.seq_length = tf.placeholder(tf.int32, [None],name="seq_length")
            # self.today = tf.placeholder(tf.int32, [None], name="today")
        # #第一层
        # layer = tf.layers.dense(
        #     inputs=self.tf_obs,
        #     units=10,
        #     activation=tf.nn.tanh,
        #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        #     bias_initializer=tf.constant_initializer(0.1),
        #     name='fc1',
        # )
        # #第二层
        # all_act = tf.layers.dense(
        #     inputs=layer,
        #     units=self.n_actions,
        #     activation=None,
        #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        #     bias_initializer=tf.constant_initializer(0.1),
        #     name='fc2'
        #
        # )

        #arg:output_dim
        basic_cell = tf.nn.rnn_cell.BasicLSTMCell(10)
        X = tf.expand_dims(self.tf_obs, axis=2)
        # print(X)
        outputs,states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
                                           sequence_length=self.seq_length, time_major=False)
        #利用softmax函数得到每个动作的概率
        c,self.h = states

        # s = tf.contrib.layers.fully_connected(
        #     self.today, 10, activation_fn=tf.nn.relu)

        self.all_act = tf.contrib.layers.fully_connected(
            self.h, 2, activation_fn = tf.nn.relu)

        self.all_act_prob = tf.nn.softmax(self.all_act, name='act_prob')
        #定义损失函数
        with tf.name_scope('loss'):
            self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.clip_by_value(self.all_act,0.001,1,name=None),
                labels=self.tf_acts)
            self.loss = tf.reduce_sum(self.neg_log_prob*self.tf_vt)
            # self.neg_log_prob = tf.cast(self.tf_acts,tf.float32) * self.all_act_prob[:,1] \
            #                      + (1 - tf.cast(self.tf_acts,tf.float32)) * self.all_act_prob[:,0]
            # self.loss = -tf.reduce_sum(tf.log(self.neg_log_prob) * self.tf_vt)
        #定义训练,更新参数
        with tf.name_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer (self.lr).minimize(self.loss)
    #定义如何选择行为，即状态ｓ处的行为采样.根据当前的行为概率分布进行采样
    def choose_action(self, observation,seq_length):
        # print(observation,seq_length)
        prob_weights = self.sess.run(self.all_act_prob,
                                     feed_dict={self.tf_obs:observation,self.seq_length:np.array([seq_length])})
        #按照给定的概率采样
        p = prob_weights.ravel()[0],prob_weights.ravel()[1]
        # print("--------------------------------")
        if p[0] > 1 - self.prob_clip:
            action = 0
        elif p[0] < self.prob_clip:
            action = 1
        else:
            action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        # print("Action Probability:", p)
        return action,p
    def greedy(self, observation,seq_length):
        prob_weights = self.sess.run(self.all_act_prob,
                                     feed_dict={self.tf_obs: observation,self.seq_length:np.array([seq_length])})
        # 按照给定的概率采样
        p = prob_weights.ravel()[0], prob_weights.ravel()[1]
        print("--------------------------------")
        print("Action Probability:", p)
        action = np.argmax(prob_weights.ravel())
        return action
    #定义存储，将一个回合的状态，动作和回报都保存在一起
    def store_transition(self, s, a, r, length):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
        self.ep_length.append(length)
    #学习，以便更新策略网络参数，一个episode之后学一回
    def learn(self):
        #计算一个episode的折扣回报
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # print(np.array(self.ep_length))
        # print(np.vstack(self.ep_obs))
        #调用训练函数更新参数
        _,loss = self.sess.run([self.train_op,self.loss], feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs_norm,
            self.seq_length: np.array(self.ep_length),
            # self.today: np.array(len(self.seq_length[-1])-1),
        })
        #清空episode数据
        # print("******************************")
        # print("Reward:",self.ep_rs,"Loss:",loss)
        # print(discounted_ep_rs_norm)
        # print(self.ep_rs)
        seq_list = copy.deepcopy(np.array(self.ep_length))
        reward_list = copy.deepcopy(self.ep_rs)
        self.ep_obs, self.ep_as, self.ep_rs,self.ep_length = [], [],[],[]
        return loss,seq_list,reward_list

    #myself reward only end get a value
    def _expand_sparse_rewards(self):
        expand_ep_rs = np.zeros_like(self.ep_rs)
        for t in range(0, len(self.ep_rs)):
            expand_ep_rs[t] = self.ep_rs[-1]

        # 归一化
        # expand_ep_rs -= np.mean(expand_ep_rs)
        # if np.std(expand_ep_rs) > 0.00001:
        #     expand_ep_rs /= np.std(expand_ep_rs)
        return expand_ep_rs

    def _discount_and_norm_rewards(self):
        #折扣回报和
        discounted_ep_rs =np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        #归一化
        # discounted_ep_rs-= np.mean(discounted_ep_rs)
        # if np.std(discounted_ep_rs) > 0.00001:
        #     discounted_ep_rs /= np.std(discounted_ep_rs)
        # print(discounted_ep_rs)
        return discounted_ep_rs