import tensorflow as tf
import numpy as np
# import gym
# import time


np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

# MAX_EPISODES = 200
# MAX_EP_STEPS = 200
# LR_A = 0.001    # learning rate for actor
# LR_C = 0.001    # learning rate for critic
# GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000

BATCH_SIZE = 32
RENDER = False
OUTPUT_GRAPH = False
# ENV_NAME = 'Pendulum-v0'

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """


    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = list(np.zeros(capacity, dtype=tuple))  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity
        self.data_pointer = 0

    def add(self, p, transition):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = transition  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.full_flag = False

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        # n就是batch size！
        # np.empty()这是一个随机初始化的一个矩阵！
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        b_memory = []
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
###############################  DDPG  #####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, LR_A=0.001, LR_C=0.001, GAMMA=0.9 ,TAU=0.01, per_memory_size=20000):
        self.pointer = 0
        self.sess = tf.Session()
        self.memory_size = MEMORY_CAPACITY
        self.memory = []
        self.per_memory = Memory(capacity=per_memory_size)
        self.per_memory_size = self.per_memory.tree.capacity
        self.pointer = 0
        self.per_pointer = 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        self.var = 0.25
        self.batch_size = BATCH_SIZE
        self.per_batch_size = BATCH_SIZE

        self.a = self._build_a(self.S, )
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            # td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.abs_errors = tf.reduce_sum(tf.abs(q_target - q), axis=1)  # for updating Sumtree
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(q_target, q))
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.loss, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):

        action_prob = self.sess.run(self.a, {self.S: np.mat(s)})[0]
        action_prob = np.clip(np.random.normal(action_prob, self.var), 0, 1)
        p = np.array([1-action_prob,action_prob])
        print(p)
        print("Var:",self.var)
        action = np.random.choice(range(2), p=p.ravel())

        return action,[1-action_prob,action_prob]

    def learn(self, per_flag=True):
        if per_flag:
            tree_idx, batch_memory, ISWeights = self.per_memory.sample(self.per_batch_size)
            batch_states, batch_actions, batch_rewards, batch_states_ = [], [], [], []
            for i in range(self.per_batch_size):
                # print(batch_memory)
                batch_states.append(batch_memory[i][0])
                batch_actions.append(batch_memory[i][1])
                batch_rewards.append(batch_memory[i][2])
                batch_states_.append(batch_memory[i][3])

            # bs = np.array(batch_states)
            # ba = np.array(batch_actions)
            # batch_rewards = np.array(batch_rewards)
            # bs_ = np.array(batch_states_)
            # br = batch_rewards[:, np.newaxis]
        else:
            bs, ba, br, bs_ = self.sample_memory()

        self.sess.run(self.atrain, {self.S: np.mat(bs)})
        _, abs_errors, cost = self.sess.run(self.ctrain, {self.S: np.mat(bs), self.a: ba, self.R: br, self.S_: np.mat(bs_),
                                                          self.ISWeights: ISWeights})

        self.per_memory.batch_update(tree_idx, abs_errors)  # update priority
        # print("lr:", self.sess.run(self.actor_lr, {self.actor_lr: actor_lr_input}))

        self.learn_step += 1


    def store_transition(self, s, a, r, s_):
        # transition = (s, a, r, s_)
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index] = transition
        # self.pointer += 1
        # print((s,a,r,s_))
        self.per_memory.store(transition=(s, a, r, s_))
        self.per_pointer = self.per_memory.tree.data_pointer
        if len(self.memory) >= self.memory_size:
            del self.memory[0]
        self.memory.append([s, a, r, s_])
        self.pointer = len(self.memory)

    def sample_memory(self):
        if len(self.memory) < self.memory_size:
            indices = np.random.choice(len(self.memory), size=self.batch_size)
        else:
            indices = np.random.choice(self.memory_size, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_states_ = [], [], [], []
        for i in indices:
            batch_states.append(self.memory[i][0])
            batch_actions.append(self.memory[i][1])
            batch_rewards.append(self.memory[i][2])
            batch_states_.append(self.memory[i][3])

        batch_states = np.array(batch_states)
        batch_actions = np.array(batch_actions)
        batch_rewards = np.array(batch_rewards)
        batch_states_ = np.array(batch_states_)
        batch_rewards = batch_rewards[:, np.newaxis]
        return batch_states, batch_actions, batch_rewards, batch_states_

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            l1 = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            # l2 = tf.layers.dense(l1, self.a_dim, activation=tf.nn.relu, name='a', trainable=trainable)
            l2 = tf.layers.dense(l1, 15, activation=tf.nn.relu, name='a', trainable=trainable)
            a = tf.layers.dense(
                inputs=l2,
                units=2,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )
            # print(tf.multiply(l2, self.a_bound, name='scaled_a'))
            # print(np.array(a[:,1:2]).shape)
            return a[:,1:2]

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)