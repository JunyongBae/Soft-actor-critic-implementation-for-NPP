import tensorflow as tf
import numpy as np

from envs.CNS_ver2 import *
from envs.myPTcurve import check_PT

state_parameter = ['PPRZ', 'ZINST63', 
                  'UAVLEGM']
min_max = [ [10e5, 170e5], [0, 100],
            [30, 350], [0, 1],
            [0, 1], [0, 1],
            [0, 1], [0, 1],
            [0, 1]]
interval = 60
end_time = 21600

record_parameter = ['PPRZ', 'ZINST63', 'UAVLEGM', 'BFV122', 'BHV142',
                   'QPRZH', 'KSWO125', 'BHV603', 'BPRZSP']

n_obs = 12
n_actions = 5

class CNS_heatup_control():

  def __init__(self, cns=1):

    self._episode_ended = False
    self.done = False

    self.cns = cns
    self.FCNS = FastCNS()
    self.target = [30.0e5, 50, 176.7] # default target

  #def action_space(self):
  #  return self._action_space

  #def observation_space(self):
  #  return self._observation_space

  def reset(self):
    self._episode_ended = False
    self.done = 0

    self.target[1] = np.random.randint(20, 60, size=1)[0]

    possible_PT = False
    while not possible_PT:
      self.target[0] = np.random.randint(20, 35, size=1)[0] * 1e5
      self.target[2] = np.random.randint(110, 180, size=1)[0]
      if check_PT(self.target[0]/1e5, self.target[2]) == 0:
        possible_PT = True

    # Phase two
    dice = np.random.rand()
    if (dice > 0.66):
      initial_condition(19, cns=self.cns)
    elif (dice > 0.33):
      initial_condition(20, cns=self.cns)
    else:
      initial_condition(21, cns=self.cns)

    integer_set('KSWO120', 1, cns=self.cns)
    integer_set('KSWO128', 1, cns=self.cns)

    real_set('BFV122', np.random.rand(), cns=self.cns)
    real_set('BHV142', np.random.rand(), cns=self.cns)

    self.one_sec()
    self.before = self.FCNS.read(state_parameter, cns=self.cns)
    self.one_sec()
    self.after = self.FCNS.read(state_parameter, cns=self.cns)
    log = self.FCNS.read(record_parameter, cns=self.cns)

    self._state = self.state(after=self.after, before=self.before)

    return np.array(self._state, dtype=np.float32), log


  def step(self, action):

    log = np.empty((0, int(len(record_parameter))))

    self.before = self.FCNS.read(state_parameter, cns=self.cns)
    for i in range(interval):
      self.action(action)
      self.one_sec()
      log = np.vstack((log, self.FCNS.read(record_parameter, cns=self.cns)))
    self.after = self.FCNS.read(state_parameter, cns=self.cns)

    # get reward and state
    reward = self.reward(after=self.after, before=self.before)
    self._state = self.state(after=self.after, before=self.before)

    self.termination(after=self.after, before=self.before)

    return self._state, np.array([reward]), np.array([self.done]), self._episode_ended, log


  def state(self, before, after):
    #state_parameter = ['PPRZ', 'ZINST63', 'UAVLEGM', 'BFV122', 'BHV142',
    #                   'QPRZH', 'KSWO125', 'BHV603', 'BPRZSP']
    #     self.target = [35.0e5, 50, 176.7] # default target

    norm_target = np.copy(self.target)

    norm_target[0] = (norm_target[0] - min_max[0][0]) / (min_max[0][1] - min_max[0][0])
    norm_target[1] = (norm_target[1] - min_max[1][0]) / (min_max[1][1] - min_max[1][0])
    norm_target[2] = (norm_target[2] - min_max[2][0]) / (min_max[2][1] - min_max[2][0])

    norm_after = self.obs_normalization(np.copy(after))
    norm_before = self.obs_normalization(np.copy(before))

    deviation = norm_after - norm_target
    change = norm_after - norm_before

    return np.hstack((norm_after, np.array(deviation), np.array(change), np.array(norm_target)))

  def obs_normalization(self, state):
    for loc, value in enumerate(state):
      state[loc] = (value - min_max[loc][0]) / (min_max[loc][1] - min_max[loc][0])

    return state


  def termination(self, after, before):

      if self.FCNS.read(['KCNTOMS'], cns=self.cns)/5 > end_time:
          self._episode_ended = True
          #self.done = 1
          print('CNS #', str(self.cns), ' : Time limit')
      #elif abs(self.FCNS.read(['UAVLEGM'], cns=self.cns) - self.target[2]) < 2:
      #    self._episode_ended = True
      #    print('CNS #', str(self.cns), ' : Success')
      elif (self.FCNS.read(['ZINST63'], cns=self.cns) > 99.0) or (self.FCNS.read(['ZINST63'], cns=self.cns) < 17.0):
          self._episode_ended = True
          #self.done = 1
          print('CNS #', str(self.cns), ' : Level Fail')
      elif check_PT(self.FCNS.read(['PPRZ'], cns=self.cns)/1e5, self.FCNS.read(['UAVLEGM'], cns=self.cns)) != 0:
          self._episode_ended = True
          #self.done = 1
          print('CNS #', str(self.cns), ' : PT Fail')
      #elif (after[2] - before[2])/interval > 28/3600:
      #    self._episode_ended = True
      #    self.done = 1
      #    print('CNS #', str(self.cns), ' : Heat up rate Limit')
      else:
          pass

  def action(self, action):


    # FV122
    FV122_position_target = (action[0] + 1) / 2
    #real_set('BFV122', FV122_position_target, cns=self.cns)
    # P controller for FV122
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
    HV142_position_target = (action[1] + 1) / 2
    #real_set('BHV142', HV142_position_target, cns=self.cns)
    # P controller for HV142
    if abs(HV142_position_target - self.FCNS.read(['BHV142'], cns=self.cns)) < 0.0376:
      integer_set('KSWO231', 0, cns=self.cns)
      integer_set('KSWO232', 0, cns=self.cns)
    elif HV142_position_target < self.FCNS.read(['BHV142'], cns=self.cns):
      integer_set('KSWO231', 1, cns=self.cns)
      integer_set('KSWO232', 0, cns=self.cns)
    else:
      integer_set('KSWO231', 0, cns=self.cns)
      integer_set('KSWO232', 1, cns=self.cns)

    PropHeater_power_target = (action[2] + 1) / 2
    real_set('QPRZH', PropHeater_power_target, cns=self.cns)

    #Backup_heater_OnOff = int(np.round((action[3] +  1) / 2))
    #Backup_heater_OnOff = int(1)
    #integer_set('KSWO125', Backup_heater_OnOff, cns=self.cns)

    HxDischarge_position_target = (action[3] + 1) / 2
    real_set('BHV603', HxDischarge_position_target, cns=self.cns)

    # Spray
    SprayValve_position_target = (action[4] + 1) / 2
    real_set('BPRZSP', SprayValve_position_target, cns=self.cns)
    # P controller for spray control
    #if abs(SprayValve_position_target - self.FCNS.read(['BPRZSP'], cns=self.cns)) < 0.0376:
    #  integer_set('KSWO126', 0, cns=self.cns)
    #  integer_set('KSWO127', 0, cns=self.cns)
    #elif SprayValve_position_target < self.FCNS.read(['BPRZSP'], cns=self.cns):
    #  integer_set('KSWO126', 1, cns=self.cns)
    #  integer_set('KSWO127', 0, cns=self.cns)
    #else:
    #  integer_set('KSWO126', 0, cns=self.cns)
    #  integer_set('KSWO127', 1, cns=self.cns)

    # KSWO128 : spray heater manual
    # KSWO126 : spray heater Toggle (Dec)
    # KSWO127 : spray heater Toggle (Inc)
    # BPRZSP : PRZ SPRAY VALVE POSITION. (0.0-1.0)


  def reward(self, after, before):

    reward = 0.1



    target_norm = np.copy(self.target)

    target_norm[0] = (target_norm[0] - min_max[0][0]) / (min_max[0][1] - min_max[0][0])
    target_norm[1] = (target_norm[1] - min_max[1][0]) / (min_max[1][1] - min_max[1][0])
    target_norm[2] = (target_norm[2] - min_max[2][0]) / (min_max[2][1] - min_max[2][0])

    before_norm = self.obs_normalization(np.copy(before))[:3]

    after_norm = self.obs_normalization(np.copy(after))[:3]


    # P, V, T rewards, respectively
    if (abs(self.FCNS.read(['PPRZ'], cns=self.cns) - self.target[0]) < 1e5): 
      reward += 10
    if (abs(self.FCNS.read(['ZINST63'], cns=self.cns) - self.target[1]) < 2):
      reward += 10
    if (abs(self.FCNS.read(['UAVLEGM'], cns=self.cns) - self.target[2]) < 3):
      reward += 10

    # Directional reward
    '''
    v = target_norm - before_norm
    optimal_direction_norm = (v) / np.sqrt(np.sum(v**2))

    reward += np.dot(optimal_direction_norm, after_norm - before_norm)
    '''
    
    # Distance reward
    #distance = tf.reduce_sum(tf.math.squared_difference(target_norm, before_norm)).numpy()
    #print(distance)
    #reward += -distance * 0.1

    #print('REAL reward : ', reward)

    
    # P, V reward
    #if (  (abs(self.FCNS.read(['PPRZ'], cns=self.cns) - self.target[0]) < 1e5) and
    #      (abs(self.FCNS.read(['ZINST63'], cns=self.cns) - self.target[1]) < 2)):
    #      reward += 1 
    # T reward
    #norm_after = self.obs_normalization(np.copy(after))
    #norm_before = self.obs_normalization(np.copy(before))

    #temp_deviation_after = abs(norm_after[2] - (self.target[2] - min_max[2][0]) / (min_max[2][1] - min_max[2][0]))
    #temp_deviation_before = abs(norm_before[2] - (self.target[2] - min_max[2][0]) / (min_max[2][1] - min_max[2][0]))

    #if temp_deviation_before > temp_deviation_after:
    #  reward = reward * 1000 * np.square(temp_deviation_before - temp_deviation_after)
    #else:
    #  reward = 0

    # Primary reward
    #if (  (abs(self.FCNS.read(['PPRZ'], cns=self.cns) - self.target[0]) < 1e5) and 
    #      (abs(self.FCNS.read(['ZINST63'], cns=self.cns) - self.target[1]) < 2) and  
    #      (abs(self.FCNS.read(['UAVLEGM'], cns=self.cns) - self.target[2]) < 3) ):
    #  reward += 10


    # Secondary reward (max = 1)
    #norm_after = self.obs_normalization(np.copy(after))

    #pressure_deviation = norm_after[0] - (self.target[0] - min_max[0][0]) / (min_max[0][1] - min_max[0][0])
    #level_deviation = norm_after[1] - (self.target[1] - min_max[1][0]) / (min_max[1][1] - min_max[1][0])
    #temp_deviation = norm_after[2] - (self.target[2] - min_max[2][0]) / (min_max[2][1] - min_max[2][0])

    #square_distance = np.square(pressure_deviation) + np.square(level_deviation) + np.square(temp_deviation)
    #reward += 0.0001 / tf.clip_by_value(square_distance, clip_value_max=1e2, clip_value_min = 0.001)

    
    # PT curve and level boundary
    if (self.FCNS.read(['ZINST63'], cns=self.cns) > 99.0) or (self.FCNS.read(['ZINST63'], cns=self.cns) < 17.0):
      reward += -10
    
    if check_PT(self.FCNS.read(['PPRZ'], cns=self.cns)/1e5, self.FCNS.read(['UAVLEGM'], cns=self.cns)) != 0:
      reward += -10

    #print('REAL reward : ', reward)


    # Heat up speed limiation
    #if error_improvement[2]/interval > 28/3600:
    #  reward += -100
    
    #error_improvement = before_error - after_error

    #pressure_weight = 1/200e5
    #level_weight = 1/100
    #temp_weight = 2/350

    #reward += (error_improvement[0] * pressure_weight
    #         + error_improvement[1] * level_weight
    #         + error_improvement[2] * temp_weight)

    # survive reward 
    #reward += 0.1

    #reward += (before_error[0] - after_error[0])/1e5



    # Directional reward 2 : Negative actions
    #if ((after_error[0] > before_error[0])
    #and (after_error[1] > before_error[1])
    #and (after_error[2] > before_error[2])):
    #  reward += -1



    '''


    # Directional reward
    before_error = np.abs(before[:3] - self.target)
    after_error = np.abs(after[:3] - self.target)
    if ((after_error[0] < before_error[0])
    and (after_error[1] < before_error[1])
    and (after_error[2] < before_error[2])):
      reward += 1

    error_improvement = before_error - after_error

    pressure_weight = 1/200e5
    level_weight = 1/100
    temp_weight = 2/350

    reward = np.sqrt(np.square(after_error[0] * pressure_weight) \
             + np.square(after_error[1] * level_weight) \
             + np.square(after_error[2] * temp_weight))

    print("Reward is",
          after_error[0] * pressure_weight,
          after_error[1] * level_weight,
          after_error[2] * temp_weight,
          reward)
    reward = 1
    # Constant reward (Maximum 10)
    if abs(self.FCNS.read(['PPRZ'], cns=self.cns) - self.target[0]) < 0.3e5:
      reward += 10
    if abs(self.FCNS.read(['ZINST63'], cns=self.cns) - self.target[1]) < 1:
      reward += 10
    if abs(self.FCNS.read(['UAVLEGM'], cns=self.cns) - self.target[2]) < 3:
      reward += 10

    # Aux. reward (Minimum -1)
    #reward += -0.00004 * np.square(self.FCNS.read(['PPRZ'], cns=self.cns)[0]/1e5 - self.target[0]/1e5)
    #print(reward)
    #reward += -0.00004 * np.square(self.FCNS.read(['ZINST63'], cns=self.cns)[0] - self.target[1])
    #print(reward)
    #reward += -0.00001 * np.square(self.FCNS.read(['UAVLEGM'], cns=self.cns)[0] - self.target[2])
    #print(reward)
    '''
    return reward

  def one_sec(self):
    self.FCNS.one_sec(cns=self.cns)