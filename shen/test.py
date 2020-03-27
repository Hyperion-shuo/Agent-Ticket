# """
# A simple version of Double Deep Q-Network(DDQN), minor change to DQN.
# To play CartPole-v0.
# Using:
# TensorFlow 2.0
# Numpy 1.16.2
# Gym 0.12.1
# """
#
# import tensorflow as tf
# print(tf.__version__)
#
# import gym
# import time
# import numpy as np
# import tensorflow.keras.layers as kl
# import tensorflow.keras.optimizers as ko
#
# np.random.seed(1)
# tf.random.set_seed(1)
#
# # Neural Network Model Defined at Here.
# class Model(tf.keras.Model):
#     def __init__(self, num_actions):
#         super().__init__(name='basic_ddqn')
#         # you can try different kernel initializer
#         self.fc1 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
#         self.fc2 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
#         self.logits = kl.Dense(num_actions, name='q_values')
#
#     # forward propagation
#     def call(self, inputs):
#         x = self.fc1(inputs)
#         x = self.fc2(x)
#         x = self.logits(x)
#         return x
#
#     # a* = argmax_a' Q(s, a')
#     def action_value(self, obs):
#         q_values = self.predict(obs)
#         best_action = np.argmax(q_values, axis=-1)
#         return best_action if best_action.shape[0] > 1 else best_action[0], q_values[0]
#
# # To test whether the model works
# def test_model():
#     env = gym.make('CartPole-v0')
#     print('num_actions: ', env.action_space.n)
#     model = Model(env.action_space.n)
#
#     obs = env.reset()
#     print('obs_shape: ', obs.shape)
#
#     # tensorflow 2.0: no feed_dict or tf.Session() needed at all
#     best_action, q_values = model.action_value(obs[None])
#     print('res of test model: ', best_action, q_values)  # 0 [ 0.00896799 -0.02111824]
#
#
# class DDQNAgent:  # Double Deep Q-Network
#     def __init__(self, model, target_model, env, buffer_size=200, learning_rate=.0015, epsilon=.1, epsilon_dacay=0.995,
#                  min_epsilon=.01, gamma=.9, batch_size=8, target_update_iter=200, train_nums=5000, start_learning=100):
#         self.model = model
#         self.target_model = target_model
#         # gradient clip
#         opt = ko.Adam(learning_rate=learning_rate, clipvalue=10.0)
#         self.model.compile(optimizer=opt, loss='mse')
#
#         # parameters
#         self.env = env                              # gym environment
#         self.lr = learning_rate                     # learning step
#         self.epsilon = epsilon                      # e-greedy when exploring
#         self.epsilon_decay = epsilon_dacay          # epsilon decay rate
#         self.min_epsilon = min_epsilon              # minimum epsilon
#         self.gamma = gamma                          # discount rate
#         self.batch_size = batch_size                # batch_size
#         self.target_update_iter = target_update_iter    # target network update period
#         self.train_nums = train_nums                # total training steps
#         self.num_in_buffer = 0                      # transition's num in buffer
#         self.buffer_size = buffer_size              # replay buffer size
#         self.start_learning = start_learning        # step to begin learning(no update before that step)
#
#         # replay buffer params [(s, a, r, ns, done), ...]
#         self.obs = np.empty((self.buffer_size,) + self.env.reset().shape)
#         self.actions = np.empty((self.buffer_size), dtype=np.int8)
#         self.rewards = np.empty((self.buffer_size), dtype=np.float32)
#         self.dones = np.empty((self.buffer_size), dtype=np.bool)
#         self.next_states = np.empty((self.buffer_size,) + self.env.reset().shape)
#         self.next_idx = 0
#
#     def train(self):
#         # initialize the initial observation of the agent
#         obs = self.env.reset()
#         for t in range(1, self.train_nums):
#             best_action, q_values = self.model.action_value(obs[None])  # input the obs to the network model
#             action = self.get_action(best_action)   # get the real action
#             next_obs, reward, done, info = self.env.step(action)    # take the action in the env to return s', r, done
#             self.store_transition(obs, action, reward, next_obs, done)  # store that transition into replay butter
#             self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)
#
#             if t > self.start_learning:  # start learning
#                 losses = self.train_step()
#                 if t % 1000 == 0:
#                     print('losses each 1000 steps: ', losses)
#
#             if t % self.target_update_iter == 0:
#                 self.update_target_model()
#             if done:
#                 obs = self.env.reset()
#             else:
#                 obs = next_obs
#
#     def train_step(self):
#         idxes = self.sample(self.batch_size)
#         s_batch = self.obs[idxes]
#         a_batch = self.actions[idxes]
#         r_batch = self.rewards[idxes]
#         ns_batch = self.next_states[idxes]
#         done_batch = self.dones[idxes]
#         # Double Q-Learning, decoupling selection and evaluation of the bootstrap action
#         # selection with the current DQN model
#         best_action_idxes, _ = self.model.action_value(ns_batch)
#         target_q = self.get_target_value(ns_batch)
#         # evaluation with the target DQN model
#         target_q = r_batch + self.gamma * target_q[np.arange(target_q.shape[0]), best_action_idxes] * (1 - done_batch)
#         target_f = self.model.predict(s_batch)
#         for i, val in enumerate(a_batch):
#             target_f[i][val] = target_q[i]
#
#         losses = self.model.train_on_batch(s_batch, target_f)
#
#         return losses
#
#     def evalation(self, env, render=True):
#         obs, done, ep_reward = env.reset(), False, 0
#         # one episode until done
#         while not done:
#             action, q_values = self.model.action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
#             obs, reward, done, info = env.step(action)
#             ep_reward += reward
#             if render:  # visually show
#                 env.render()
#             time.sleep(0.05)
#         env.close()
#         return ep_reward
#
#     # store transitions into replay butter
#     def store_transition(self, obs, action, reward, next_state, done):
#         n_idx = self.next_idx % self.buffer_size
#         self.obs[n_idx] = obs
#         self.actions[n_idx] = action
#         self.rewards[n_idx] = reward
#         self.next_states[n_idx] = next_state
#         self.dones[n_idx] = done
#         self.next_idx = (self.next_idx + 1) % self.buffer_size
#
#     # sample n different indexes
#     def sample(self, n):
#         assert n < self.num_in_buffer
#         res = []
#         while True:
#             num = np.random.randint(0, self.num_in_buffer)
#             if num not in res:
#                 res.append(num)
#             if len(res) == n:
#                 break
#         return res
#
#     # e-greedy
#     def get_action(self, best_action):
#         if np.random.rand() < self.epsilon:
#             return self.env.action_space.sample()
#         return best_action
#
#     # assign the current network parameters to target network
#     def update_target_model(self):
#         self.target_model.set_weights(self.model.get_weights())
#
#     def get_target_value(self, obs):
#         return self.target_model.predict(obs)
#
#     def e_decay(self):
#         self.epsilon *= self.epsilon_decay
#
# if __name__ == '__main__':
#     test_model()
#
#     env = gym.make("CartPole-v0")
#     num_actions = env.action_space.n
#     model = Model(num_actions)
#     target_model = Model(num_actions)
#     agent = DDQNAgent(model, target_model, env)
#     # test before
#     rewards_sum = agent.evalation(env)
#     print("Before Training: %d out of 200" % rewards_sum) # 9 out of 200
#
#     agent.train()
#     # test after
#     # env = gym.wrappers.Monitor(env, './recording', force=True)  # to record the process
#     rewards_sum = agent.evalation(env)
#     print("After Training: %d out of 200" % rewards_sum) # 200 out of 200

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import RMSprop

from DQN.maze_env import Maze


class Eval_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network')
        self.layer1 = layers.Dense(10, activation='relu')
        self.logits = layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class Target_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network_1')
        self.layer1 = layers.Dense(10, trainable=False, activation='relu')
        self.logits = layers.Dense(num_actions, trainable=False, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class DeepQNetwork:
    def __init__(self, n_actions, n_features, eval_model, target_model):

        self.params = {
            'n_actions': n_actions,
            'n_features': n_features,
            'learning_rate': 0.01,
            'reward_decay': 0.9,
            'e_greedy': 0.9,
            'replace_target_iter': 300,
            'memory_size': 500,
            'batch_size': 32,
            'e_greedy_increment': None
        }

        # total learning step

        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.epsilon = 0 if self.params['e_greedy_increment'] is not None else self.params['e_greedy']
        self.memory = np.zeros((self.params['memory_size'], self.params['n_features'] * 2 + 2))

        self.eval_model = eval_model
        self.target_model = target_model

        self.eval_model.compile(
            optimizer=RMSprop(lr=self.params['learning_rate']),
            loss='mse'
        )
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.params['memory_size']
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_model.predict(observation)
            print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.params['n_actions'])
        return action

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.params['memory_size']:
            sample_index = np.random.choice(self.params['memory_size'], size=self.params['batch_size'])
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.params['batch_size'])

        batch_memory = self.memory[sample_index, :]

        q_next = self.target_model.predict(batch_memory[:, -self.params['n_features']:])
        q_eval = self.eval_model.predict(batch_memory[:, :self.params['n_features']])

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.params['batch_size'], dtype=np.int32)
        eval_act_index = batch_memory[:, self.params['n_features']].astype(int)
        reward = batch_memory[:, self.params['n_features'] + 1]

        q_target[batch_index, eval_act_index] = reward + self.params['reward_decay'] * np.max(q_next, axis=1)

        # check to replace target parameters
        if self.learn_step_counter % self.params['replace_target_iter'] == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
            print('\ntarget_params_replaced\n')

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network

        self.cost = self.eval_model.train_on_batch(batch_memory[:, :self.params['n_features']], q_target)

        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.params['e_greedy_increment'] if self.epsilon < self.params['e_greedy'] \
            else self.params['e_greedy']
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
            step += 1
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    eval_model = Eval_Model(num_actions=env.n_actions)
    target_model = Target_Model(num_actions=env.n_actions)
    RL = DeepQNetwork(env.n_actions, env.n_features, eval_model, target_model)
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()