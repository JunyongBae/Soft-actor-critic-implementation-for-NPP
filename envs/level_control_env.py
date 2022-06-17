import abc
import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Box

from envs.CNS_ver2 import *
from envs.myPTcurve import check_PT

tf.compat.v1.enable_v2_behavior()

state_parameter = ['ZINST63','BFV122', 'BHV142']
interval = 60
end_time = 4200

record_parameter = ['ZINST63', 'PPRZ', 'BPRZSP', 'BFV122', 'BHV142']


class CNS_level_control(gym.Env):

  def __init__(self, cns=1):
    self.action_space = Box(low=0, high=1, shape=(2, ), dtype=np.float32)
    self.observation_space = Box(low=0, high=100, shape=(5, ), dtype=np.float32)

    self._episode_ended = False
    self.done = False

    self.cns = cns
    self.FCNS = FastCNS()
    self.target = 50  # default target

  #def action_space(self):
  #  return self._action_space

  #def observation_space(self):
  #  return self._observation_space

  def reset(self):
    self._episode_ended = False
    self.done = False
    self.target = np.random.randint(30, 70, size=1)[0]

    self.log = np.empty((0, int(len(record_parameter))))

    # Phase two
    if (np.random.rand() > 0.5):
      initial_condition(19, cns=self.cns)
    else:
      initial_condition(20, cns=self.cns)

    real_set('BFV122', np.random.rand(), cns=self.cns)
    real_set('BHV142', np.random.rand(), cns=self.cns)

    self.one_sec()
    self.after = self.FCNS.read(state_parameter, cns=self.cns)
    self.log = np.vstack((self.log, self.FCNS.read(record_parameter, cns=self.cns)))

    self._state = self.state(after=self.after)

    return np.array(self._state, dtype=np.float32)


  def step(self, action):


    for i in range(interval):
      self.action(action)
      self.spray()
      self.one_sec()
      self.log = np.vstack((self.log, self.FCNS.read(record_parameter, cns=self.cns)))
    self.after = self.FCNS.read(state_parameter, cns=self.cns)

    # get reward and state
    reward = self.reward()
    self._state = self.state(after=self.after)

    self.termination()

    return self._state, reward, self.done, self._episode_ended


  def state(self, after):

    lvl_deviation = after[0] - self.target

    return np.hstack((after, np.array([lvl_deviation, self.target])))


  def termination(self):

      if self.FCNS.read(['KCNTOMS'], cns=self.cns)/5 > end_time:
          self._episode_ended = True
          print('CNS #', str(self.cns), ' : Finish')
      elif (self.FCNS.read(['ZINST63'], cns=self.cns) > 99.0) or (self.FCNS.read(['ZINST63'], cns=self.cns) < 17.0):
          self._episode_ended = True
          self.done = True
          print('CNS #', str(self.cns), ' : Level Fail')
      elif check_PT(self.FCNS.read(['PPRZ'], cns=self.cns)/1e5, self.FCNS.read(['UAVLEGM'], cns=self.cns)) != 0:
          self._episode_ended = True
          self.done = True
          print('CNS #', str(self.cns), ' : PT Fail')
      else:
          pass

  def action(self, action):

    # FV122
    # real_set('BFV122', action[0], cns=self.cns)
    # P controller of FV122
    FV122_position_target = (action[0] + 1) / 2
    if abs(FV122_position_target - self.FCNS.read(['BFV122'], cns=self.cns)) < 0.0376:
      integer_set('KSWO101', 0, cns=self.cns)
      integer_set('KSWO102', 0, cns=self.cns)
    elif FV122_position_target < self.FCNS.read(['BFV122'], cns=self.cns):
      integer_set('KSWO101', 1, cns=self.cns)
      integer_set('KSWO102', 0, cns=self.cns)
    else:
      integer_set('KSWO101', 0, cns=self.cns)
      integer_set('KSWO102', 1, cns=self.cns)

    # HV142
    # real_set('BHV142', action[1], cns=self.cns)
    # P controller of HV142
    HV142_position_target = (action[1] + 1) / 2
    if abs(HV142_position_target - self.FCNS.read(['BHV142'], cns=self.cns)) < 0.0376:
      integer_set('KSWO231', 0, cns=self.cns)
      integer_set('KSWO232', 0, cns=self.cns)
    elif HV142_position_target < self.FCNS.read(['BHV142'], cns=self.cns):
      integer_set('KSWO231', 1, cns=self.cns)
      integer_set('KSWO232', 0, cns=self.cns)
    else:
      integer_set('KSWO231', 0, cns=self.cns)
      integer_set('KSWO232', 1, cns=self.cns)

    # Spray
    # KSWO128 : spray heater manual
    # KSWO126 : spray heater Toggle (Dec)
    # KSWO127 : spray heater Toggle (Inc)
    # BPRZSP : spray heater power (%)

  def spray(self):
    press = self.FCNS.read(['PPRZ'], cns=self.cns)
    if press > 28e5:
      integer_set('KSWO126', 0, cns=self.cns)
      integer_set('KSWO127', 1, cns=self.cns)
    elif press < 22e5:
      integer_set('KSWO126', 1, cns=self.cns)
      integer_set('KSWO127', 0, cns=self.cns)
    else:
      integer_set('KSWO126', 0, cns=self.cns)
      integer_set('KSWO127', 0, cns=self.cns)

  def reward(self):

    reward = 0
    # Constant reward (Maximum 10)
    if abs(self.FCNS.read(['ZINST63'], cns=self.cns) - self.target) < 1:
      reward += 10
    else:
      reward += 0

    # Aux. reward (Minimum -1)
    reward += -0.0004 * np.square(self.FCNS.read(['ZINST63'], cns=self.cns)[0] - self.target)

    return reward

  def one_sec(self):
    self.FCNS.one_sec(cns=self.cns)