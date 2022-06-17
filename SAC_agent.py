from asyncio import base_tasks
from queue import Queue
from random import sample
from threading import Thread

from numpy import dtype
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

from SAC_episode import Episode
from SAC_replay_buffer import ReplayBuffer
from envs.heat_up_env import n_obs, n_actions
from SAC_network import Actor, Critic

class Train():

    def __init__(self) -> None:
        # Constant
        self.buffer_size = 2000000
        self.batch_size = 256
        self.actor_lr = 0.00001
        self.critic_lr = 0.00001
        self.temp_lr = 0.000005
        self.temp = tf.Variable(0.5)
        self.gamma = 0.99
        self.tau = 0.001
        self.entropy_target = -n_actions

        # Shared replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size = self.buffer_size)

        # Multiple CNS setting
        self.waiting_line = Queue()
        self.n_cns = 10
        for i in range(self.n_cns):
            self.waiting_line.put(Episode(waiting_line=self.waiting_line,
                                            replay_buffer=self.replay_buffer,
                                            cns=i+1))

        # Main Actor and Critic networks
        self.main_actor = Actor(n_actions)
        self.main_actor.build(input_shape = (None, n_obs))
        self.actor_optimizer = Adam(self.actor_lr)

        self.critic = Critic()
        self.target_critic = Critic()
        input_state = Input((n_obs, ))
        input_action = Input((n_actions, ))
        self.critic(input_state, input_action)
        self.target_critic(input_state, input_action)
        self.critic_optimizer = Adam(self.critic_lr)

        self.temp_optimizer = Adam(self.temp_lr)

    def update_target_critic(self, tau):
        theta = self.critic.get_weights()
        tareget_theta = self.target_critic.get_weights()

        for i in range(len(theta)):
            tareget_theta[i] = tau * theta[i] + (1 - tau) * tareget_theta[i]

        self.target_critic.set_weights(tareget_theta)


    def get_q_targets(self, next_states, rewards, dones):
        next_mu, next_std = self.main_actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
        next_actions, next_log_pdfs = self.main_actor.sample_normal(next_mu, next_std)
        
        q_next = self.target_critic(next_states, next_actions)

        q_targets = rewards + (1 - dones) * (self.gamma * (q_next - self.temp * next_log_pdfs)).numpy()

        return q_targets

    def critic_learn(self, states, actions, q_targets):
        with tf.GradientTape() as tape:
            q = self.critic(states, actions, training=True)
            loss = tf.reduce_mean(tf.square(q-q_targets))

        print('Critic loss :', loss.numpy())

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            mu, std = self.main_actor(states, training=True)
            actions, log_pdfs = self.main_actor.sample_normal(mu, std)
            soft_q = self.critic(states, actions)

            loss = tf.reduce_mean(self.temp * log_pdfs - soft_q)

        print('Actor loss : ', loss.numpy())

        grads = tape.gradient(loss, self.main_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.main_actor.trainable_variables))

    def temp_learn(self, states):

        with tf.GradientTape() as tape:
            mu, std = self.main_actor(states, training=True)
            _, log_pdfs = self.main_actor.sample_normal(mu, std)

            loss = tf.reduce_mean(-self.temp * log_pdfs - self.temp * self.entropy_target)
        
        grads = tape.gradient(loss, [self.temp])
        #print('Temperature : ', self.temp.numpy())
        self.temp_optimizer.apply_gradients(zip(grads, [self.temp]))




    def train(self):
        max_episode_num = 10000

        self.update_target_critic(1.0)

        for n_trial in range(max_episode_num):

            ep = self.waiting_line.get()

            print(  'Finish Trial # ' + str(ep.n_trial) + 
                    ': Reward = ' + str(ep.episode_reward) +
                    ', Episode len = ' + str(ep.episode_len))

            itr = np.copy(ep.episode_len)

            # copy out the wights of main actor
            ep.actor.set_weights(self.main_actor.get_weights())
            Thread(target=ep.trial, args=(n_trial,)).start()

            # Trainning
            if self.replay_buffer.num_experiences > self.batch_size:
                for _ in range(itr * 2):
                    sample = self.replay_buffer.get_batch(self.batch_size)

                    # Critic training
                    q_targets = self.get_q_targets(sample["states1"], sample["rewards"], sample["terminals1"])
                    self.critic_learn(sample["states0"], sample["actions"], q_targets)

                    # Actor traininig
                    self.actor_learn(sample["states0"])

                    # Temp. traininig
                    self.temp_learn(sample["states0"])

                    # update tareget critic slowly
                    self.update_target_critic(self.tau)

            self.main_actor.save_weights('./models/actor_' + str(n_trial) + '.h5')
            self.critic.save_weights('./models/critic_' + str(n_trial) + '.h5')




main = Train()
main.train()