import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate



# This is actor network
class Actor(Model):

    def __init__(self, action_dim) -> int:
        super(Actor, self).__init__()

        self.action_dim = action_dim
        #self.std_bound = [1e-6, 4.0]
        
        self.h1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.h2 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.h3 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.mu = Dense(action_dim, activation='linear')
        self.std = Dense(action_dim, activation='softplus')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)
        std = self.std(x)

        #std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])

        return mu, std

    def sample_normal(self, mu, std):
        normal_dist = tfp.distributions.Normal(mu, std)

        # Enforcing Action bound (See SAC paper)
        sampled_mu = normal_dist.sample()
        action = tf.tanh(sampled_mu)
        log_pdf = (tf.reduce_sum(normal_dist.log_prob(sampled_mu), 1, keepdims=True) 
                    - tf.reduce_sum(tf.math.log(1.0 - tf.pow(action, 2) + 1e-6), 1, keepdims=True))
                    # 1e-6 : numerical stability for log calculation
        
        return action, log_pdf


# This is critic network
class Critic(Model):

    def __init__(self):
        super(Critic, self).__init__()

        self.state_layer = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.action_layer = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.h1 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.h2 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.h3 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.h4 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.value = Dense(1)


    def call(self, state, action):

        s = self.state_layer(state)
        a = self.action_layer(action)
        h = concatenate([s, a], axis=-1)
        x = self.h1(h)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        value = self.value(x)

        return value


'''
actor = Actor(2)
actor.build(input_shape=(None, 3))

critic = Critic()
input_state = Input((3, ))
input_action = Input((2, ))
critic(input_state, input_action)


test = np.array([[15, 0.7, 33], [-15, 0.7, 33], [-15, -0.7, 33]])
mu, std = actor(tf.convert_to_tensor(test, dtype=tf.float64))
action, log_pdf = actor.sample_normal(mu, std)
print(mu, std)
print(action, log_pdf)

value = critic(test, action)

print('this is state', test)
print('this is action', action)

print('this is value', value)
'''