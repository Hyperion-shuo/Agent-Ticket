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
        # 启动一个默认的会话
        self.sess = tf.Session()
        #创建策略网络
        # self._build_net()
        self.actor = Actor(self.sess, n_features=n_features, n_actions=n_actions, lr=LR_A)
        self.critic = Critic(self.sess, n_features=n_features, lr=LR_C)#据说critic作为评价网络，学习率要大于actor

        self.prob_clip = prob_clip


        # if output_graph:
        #     tf.summary.FileWriter("logs/", self.sess.graph)
        # 初始化会话中的变量
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state, length):
        return self.actor.choose_action(state, length)

    def criticLearn(self, state, reward, state_):
        return self.critic.learn(state, reward, state_)

    def actorLearn(self, state, action, td_error):
        return self.actor.learn(state, action, td_error)

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(tf.clip_by_value(self.acts_prob[0, self.a],0.000001,1,name=None))
            self.exp_v = tf.reduce_sum(log_prob * self.td_error)  # advantage (TD_error) guided loss
            # self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s, length):
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        # print(probs)
        action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
        p = probs.ravel()[0],probs.ravel()[1]
        print(p)
        return action,p


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, gamma=0.9):
        self.sess = sess
        self.gamma = gamma
        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, None, "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')


        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
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

    def learn(self, s, r, s_):

        if s_[0] == -1 and len(s_) == 1:
            v_ = np.mat([0])
        else:
            s_ = np.mat(s_)
            # s_ = s_[np.newaxis, :]
            v_ = self.sess.run(self.v, {self.s: s_})
            # print("V(S):",v_)



        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error