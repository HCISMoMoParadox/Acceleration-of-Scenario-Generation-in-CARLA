import random
import numpy as np
from cem.init_conds import *
import zmq
import time

if __name__ == '__main__':
    # peak 11 valley 15
    i = 15
    data = np.load('./cem/checkpoint/3_t1-2_0_m_f_l_1_0/cem_iter=%d.npy' % i, allow_pickle=True)
    data = data.tolist()
    nat_param_wp = data['nat_param_wp']
    nat_param_w = data['nat_param_w']
    print('nat_param_wp', nat_param_wp)
    print('nat_param_w', nat_param_w)
    
    sample_size = 1
    # sample
    base_wp=np.array([[2.0, 2.0]])
    base_w=np.array([[2.0,2.0]])
    print('base_wp')
    print(base_wp)
    print('base_w')
    print(base_w)
    ic=init_conds_test(base_wp,base_w)

    obs_wp=ic.sample_obs(ic.model_wp,nat_param_wp,sample_size)
    obs_w=ic.sample_obs(ic.model_w,nat_param_w,sample_size)
    sample_wp = (obs_wp * 10).astype(int)
    sample_w = (obs_w * 120.0) - 60.0

    # random
    # sample_size = 100
    # sample_wp = (np.random.rand(sample_size) * 10).astype(int)
    # sample_w = (np.random.rand(sample_size)  * 120.0) - 60.0

    #  Socket to talk to server
    context = zmq.Context()
    print("Connecting to worker server…")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    # socket.connect("tcp://140.113.240.44:5555")
    collided_list = list()
    TTC_list = list()
    for j in range(sample_size):
      param1 = sample_wp[0][j]
      param2 = sample_w[0][j]
      #random
    #   param1 = sample_wp[j]
    #   param2 = sample_w[j]

      
      start_time = time.time()
      print("Sending request iteration %d …" % j)
      print(f'param {param1}, {param2}')
      # send a task to a worker
      socket.send_string(("%d,%d" % (param1, param2)))

      #  Get the reply.
      message = socket.recv()
      elapse_time = time.time() - start_time
      print("Received reply [ %s ]" % (message))
      encoding = 'utf-8'
      message = message.decode(encoding)
      collided_str = message.split(',')[0]
      TTC_str = message.split(',')[1]
      collided_list.append(1 if collided_str == 'True' else 0)
      TTC_list.append(float(TTC_str))
      
      print('elapse_time:', elapse_time)
    np.save('TTC_list_923.npy', TTC_list)
    np.save('collided_list_923.npy', collided_list)
    print('collision rate:', np.sum(collided_list)/100.0)
    print('average TTC:', np.mean(TTC_list))
