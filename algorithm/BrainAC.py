import numpy as np
import tensorflow as tf
import copy
np.random.seed(1)
tf.set_random_seed(1)

class ActorCritic():
    def __init__(
            self,
            n_actions=2,
            n_features=87,
            LR_A = 0.001, # learning rate for actor
            LR_C = 0.01,  # learning rate for critic
            reward_decay=0.95,
            prob_clip=0.06,
            RNN_num = 1,
            output_graph=False,
    ):
        #动作空间的维数
        self.n_actions = n_actions
        #状态特征的维数
        self.n_features = n_features
        #回报衰减率
        self.gamma = reward_decay
        #一条轨迹的观测值，动作值，和回报值
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]
        self.ep_length = []
        self.RNN_num = RNN_num
        # 启动一个默认的会话
        self.sess = tf.Session()
        #创建策略网络
        # self._build_net()
        self.actor = Actor(self.sess, n_features=n_features, n_actions=n_actions, lr=LR_A, RNN_num=self.RNN_num)
        self.critic = Critic(self.sess, n_features=n_features, lr=LR_C, RNN_num=self.RNN_num)#据说critic作为评价网络，学习率要大于actor

        self.prob_clip = prob_clip



        # if output_graph:
        #     tf.summary.FileWriter("logs/", self.sess.graph)
        # 初始化会话中的变量
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state, length):
        stateP = self.statePreprocess2(state)
        return self.actor.choose_action(stateP, length)

    def criticLearn(self, state, reward, state_, length):
        stateP = self.statePreprocess2(state)
        state_P = self.statePreprocess2(state_)
        return self.critic.learn(stateP, reward, state_P, length)

    def actorLearn(self, state, action, td_error, length):
        stateP = self.statePreprocess2(state)
        return self.actor.learn(stateP, action, td_error, length)

    def statePreprocess(self,state):
        exist = np.array([n for n in state if n>0])
        exist -= 1877.368
        exist /= 256.61
        exist = exist.tolist()
        while len(exist) < 87:
            exist.append(0)
        # print("STATE:", self.state,"EXIST:",exist)
        return exist

    def statePreprocess2(self,state):
        processed = []
        for i in range(len(state)):
            exist = np.array([n for n in state[i] if n>0])
            exist -= 1877.368
            exist /= 256.61
            exist = exist.tolist()
            while len(exist) < 87:
                exist.append(0)
            processed.append(exist)
            # print("STATE:", self.state,"EXIST:",exist)
        return processed

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001, RNN_num=1):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, RNN_num,n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.seq_length = tf.placeholder(tf.int32, [None], name="seq_length")

        with tf.variable_scope('Actor'):
            # l1 = tf.layers.dense(
            #     inputs=self.s,
            #     units=20,    # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l1'
            # )
            rnn_outputs = []
            basic_cell = tf.nn.rnn_cell.BasicLSTMCell(10)
            for i in range(RNN_num):
                X = self.s[0][i]
                X = X[np.newaxis ,:]
                X = tf.expand_dims(X, axis=2)
                _, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
                                                    sequence_length=self.seq_length, time_major=False)
                # 利用softmax函数得到每个动作的概率
                outputs, self.h = states
                rnn_outputs.append(outputs)
            rnn_outputs = tf.concat(rnn_outputs, 1)

            self.acts_prob = tf.layers.dense(
                inputs=rnn_outputs,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            # print(self.a)
            log_prob = tf.log(tf.clip_by_value(self.acts_prob[0, self.a],0.000001,1,name=None))
            self.exp_v = tf.reduce_sum(log_prob * self.td_error)  # advantage (TD_error) guided loss
            # self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td, l):
        s = np.array(s)
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td, self.seq_length:np.array([l])}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s, length):
        s = np.array(s)
        s = s[np.newaxis,:]
        probs = self.sess.run(self.acts_prob, {self.s: s,self.seq_length:np.array([length])})   # get probabilities for all actions
        # print(probs)
        action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
        p = probs.ravel()[0],probs.ravel()[1]
        print(p)
        return action,p


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, gamma=0.9, RNN_num=1):
        self.sess = sess
        self.gamma = gamma
        self.s = tf.placeholder(tf.float32, [None, RNN_num, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, None, "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.seq_length = tf.placeholder(tf.int32, [None], name="seq_length")


        with tf.variable_scope('Critic'):
            # l1 = tf.layers.dense(
            #     inputs=self.s,
            #     units=20,  # number of hidden units
            #     activation=tf.nn.relu,  # None
            #     # have to be linear to make sure the convergence of actor.
            #     # But linear approximator seems hardly learns the correct Q.
            #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l1'
            # )
            rnn_outputs = []
            basic_cell = tf.nn.rnn_cell.BasicLSTMCell(10)
            for i in range(RNN_num):
                X = self.s[0][i]
                X = X[np.newaxis, :]
                X = tf.expand_dims(X, axis=2)
                _, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
                                              sequence_length=self.seq_length, time_major=False)
                # 利用softmax函数得到每个动作的概率
                outputs, self.h = states
                rnn_outputs.append(outputs)
            rnn_outputs = tf.concat(rnn_outputs, 1)

            self.v = tf.layers.dense(
                inputs=rnn_outputs,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + self.gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_, l):

        if s_[0] == -1 and len(s_) == 1:
            v_ = np.mat([0])
        else:
            s_ = np.array(s_)
            s_ = s_[np.newaxis, :]
            # s_ = s_[np.newaxis, :]
            v_ = self.sess.run(self.v, {self.s: s_, self.seq_length:np.array([l+1])})
            # print("V(S):",v_)
        s = np.array(s)
        s = s[np.newaxis, :]


        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r, self.seq_length:np.array([l])})
        return td_error