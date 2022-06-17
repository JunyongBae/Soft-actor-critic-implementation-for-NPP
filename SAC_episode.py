import tensorflow as tf
import numpy as np
import csv
from collections import deque

from SAC_network import Actor
from envs.heat_up_env import CNS_heatup_control, record_parameter, n_obs, n_actions

class Episode(object):

    def __init__(self, waiting_line, replay_buffer, cns = 1):
        self.waiting_line = waiting_line
        self.replay_buffer = replay_buffer
        
        self.train_env = CNS_heatup_control(cns = cns)
        self.n_obs = n_obs
        self.n_acitons = n_actions

        self.actor = Actor(self.n_acitons)
        self.actor.build(input_shape = (None, self.n_obs))

        self.episode_len = 0
        self.episode_reward = 0
        self.n_trial = 0

    def batch_action(self, state):
        mu, std = self.actor(tf.convert_to_tensor(np.array([[state]]), dtype=tf.float64))
        action, _ = self.actor.sample_normal(mu, std)

        return tf.squeeze(action).numpy()

    
    def trial(self, n_trial):
        


        # setting for HER - future strategy
        HER_margin = 2
        HER_future = 3 + (HER_margin)
        trajectory = deque(maxlen=HER_future)

        # Reset
        state, log = self.train_env.reset()

        self.episode_len = 0
        self.episode_reward = 0
        self.n_trial = n_trial

        # Logger
        log_file = open('./record/trial_' + str(n_trial)
                         + '_' + str(self.train_env.target) +'.csv', 'w', newline='')
        logger = csv.writer(log_file)
        logger.writerow(record_parameter)
        logger.writerow(np.round(log, 3))

        episode_ended = False
        while not episode_ended:

            action = self.batch_action(state=state)
            new_state, reward, done, episode_ended, log = self.train_env.step(action)

            # Replay buffer
            self.replay_buffer.add(state, action, reward, new_state, done)

            # setting for HER - future strategy
            trajectory.append((state, action, new_state, done))


            if int(len(trajectory)) == HER_future and (np.random.random() < 0.0):
                #print('Reviewed state :', trajectory[0][0][:9])
                #print('Reviewed new state :', trajectory[0][2][:9])
                #print('HER tareget : ', trajectory[-1][0][:3])
                #print('Deviation state : ', trajectory[0][0][:3] - trajectory[-1][0][:3])
                #print('Deviation new state : ', trajectory[0][2][:3] - trajectory[-1][0][:3])

                HER_state = np.hstack((trajectory[0][0][:3], 
                                        trajectory[0][0][:3] - trajectory[-HER_margin][0][:3],
                                        trajectory[0][0][6:9], 
                                        trajectory[-HER_margin][0][:3]))
                #print('HER state : ', HER_state)
                
                HER_new_state = np.hstack((trajectory[0][2][:3], 
                                            trajectory[0][2][:3] - trajectory[-HER_margin][0][:3],
                                            trajectory[0][2][6:9], 
                                            trajectory[-HER_margin][0][:3]))
                #print('HER new state : ', HER_new_state)
                
                HER_action = trajectory[0][1]
                #print('HER action : ', HER_action)

                HER_done = trajectory[0][3]
                #print('HER done : ', HER_done)

                HER_reward = 0.1

                target_norm = trajectory[-HER_margin][0][:3]

                before_norm = trajectory[0][0][:3]

                after_norm = trajectory[0][2][:3]
                '''
                v = target_norm - before_norm
                optimal_direction_norm = (v) / np.sqrt(np.sum(v**2))

                HER_reward += np.dot(optimal_direction_norm, after_norm - before_norm)
                '''
                #distance = tf.reduce_sum(tf.math.squared_difference(target_norm, before_norm)).numpy()
                #print(distance)     
                #HER_reward += -distance * 0.1

                if (abs(trajectory[0][2][0] - trajectory[-HER_margin][0][0]) < (1e5 / (170e5 - 10e5))):
                    HER_reward += 10
                if (abs(trajectory[0][2][1] - trajectory[-HER_margin][0][1]) < (2 / (100 - 0))):
                    HER_reward += 10
                if (abs(trajectory[0][2][2] - trajectory[-HER_margin][0][2]) < (3 / (350 - 30))): 
                    HER_reward += 10


                #if ((abs(trajectory[0][2][0] - trajectory[-HER_margin][0][0]) < (1e5 / (170e5 - 10e5))) and
                #    (abs(trajectory[0][2][1] - trajectory[-HER_margin][0][1]) < (2 / (100 - 0))) and
                #    (abs(trajectory[0][2][2] - trajectory[-HER_margin][0][2]) < (3 / (350 - 30)))): 
                #    HER_reward += 10


                '''
                temp_deviation_after = abs(trajectory[0][2][2] - trajectory[-1][0][2])
                temp_deviation_before = abs(trajectory[0][0][2] - trajectory[-1][0][2])
                if temp_deviation_before > temp_deviation_after:
                    HER_reward = HER_reward * 1000 * np.square(temp_deviation_before - temp_deviation_after)
                else:
                    HER_reward = 0
                print('HER reward : ', HER_reward)
                '''

                #print('HER reward : ', HER_reward )

                self.replay_buffer.add(HER_state, HER_action, HER_reward, HER_new_state, HER_done)
          



            # Logger
            logger.writerows(np.round(log, 3))

            state = new_state

            self.episode_len += 1
            self.episode_reward += reward[0]

        log_file.close()
        self.waiting_line.put(self)



