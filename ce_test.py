import sys
sys.path.append('./cem/')

from init_conds import *
from ce_log import *
import glob
import os

# try:
    #
sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

def test_objective(obs_v,obs_w,obs_x,obs_y,obs_gail,sockets=0,ic=0):
 return .5*(np.sum(np.power(obs_v,2.)+np.power(obs_w,2.)+np.power(obs_x,2.)+np.power(obs_y,2.),0)+np.sum(np.power(obs_gail,2.),0))
def test_objective_simple(obs_v,obs_w,obs_x,obs_y,obs_gail,sockets=0,ic=0):
 return obs_x[0,:]
def test_gradient(obs_v,obs_w,obs_x,obs_y,obs_gail,sockets=0):
 return obs_v,obs_w,obs_x,obs_y,obs_gail
def test_ce(seed_value):
 np.random.seed(seed_value)
 num_agents=2
 base_v=np.abs(np.random.randn(num_agents,2))
 base_w=np.array([[1.8,4.1],[2.1,2.8]])
 base_x=np.abs(np.column_stack((np.random.randn(num_agents),10*np.random.rand(num_agents))))
 base_y=np.array([[1.2,2.3],[3.1,1.8]])
 print('base_v')
 print(base_v)
 print('base_w')
 print(base_w)
 print('base_x')
 print(base_x)
 print('base_y')
 print(base_y)
 dim_gail=20
 mu=5.*np.random.randn(dim_gail)
 sigma=np.diag(np.random.rand(dim_gail))
 base_gail=np.vstack((mu,sigma))
 print('gail mean')
 print(mu)
 lanes=np.array([4,5])
 ic=init_conds.init_conds(base_v,base_w,base_x,base_y,base_gail,lanes)
 base_mean=np.power(base_v,2.).sum()+2.*num_agents +\
  np.power(base_x[:,0],2.).sum() +np.power(mu,2.).sum()
 levels_mult=np.array([1.,3.,5.,10.])
 print('levels')
 print(levels_mult*base_mean)
 sample_size=int(1000000.)
 num_iter=100
 save_iter=(np.array([30,70,100])-1).astype(int)
 rho=.95
 alpha=.9
 ce_sample_size=5000
 ns_naive=sample_size+int(ce_sample_size*(save_iter[-1]+1))
 obs_v=ic.sample_obs(ic.model_v,ic.nat_base_v,ns_naive)
 obs_w=ic.sample_obs(ic.model_w,ic.nat_base_w,ns_naive)
 obs_x=ic.sample_obs(ic.model_x,ic.nat_base_x,ns_naive)
 obs_y=ic.sample_obs(ic.model_y,ic.nat_base_y,ns_naive)
 obs_gail=ic.sample_obs(ic.model_gail,ic.nat_base_gail,ns_naive,if_gail=True)
 objs_naive=test_objective(obs_v,obs_w,obs_x,obs_y,obs_gail,0.)
 objs_ce=np.empty((save_iter.size,sample_size))
 like_ratio=np.empty((save_iter.size,sample_size))
 sample_sizes=np.full(num_iter,ce_sample_size).astype(int)
 step_sizes=np.full(num_iter,alpha)
 all_nat_param_v,all_nat_param_w,all_nat_param_x, all_nat_param_y,\
  all_nat_param_gail = cross_entropy(ic,rho,levels_mult[-1]*base_mean,\
    num_iter,save_iter, sample_sizes,step_sizes,test_objective)
 counter=0
 for num_iter in save_iter:
  print('----------------------- Processing '+str(num_iter+1) +' -------------------')
  obs_v=ic.sample_obs(ic.model_v,all_nat_param_v[counter,:],sample_size)
  obs_w=ic.sample_obs(ic.model_w,all_nat_param_w[counter,:],sample_size)
  obs_x=ic.sample_obs(ic.model_x,all_nat_param_x[counter,:],sample_size)
  obs_y=ic.sample_obs(ic.model_y,all_nat_param_y[counter,:],sample_size) 
  obs_gail=ic.sample_obs(ic.model_gail,all_nat_param_gail[counter,:],sample_size,if_gail=True)
  objs_ce[counter,:]=test_objective(obs_v,obs_w,obs_x,obs_y,obs_gail,0.)
  print('mean_ce')
  print(np.mean(objs_ce[counter,:]))
  like_ratio_v,like_ratio_w,like_ratio_x,like_ratio_y,\
    like_ratio_gail =ic.compute_like_ratio(obs_v,obs_w,\
      obs_x,obs_y,obs_gail,all_nat_param_v[counter,:],\
        all_nat_param_w[counter,:],all_nat_param_x[counter,:],\
          all_nat_param_y[counter,:],all_nat_param_gail[counter,:])
  like_ratio[counter,:]=like_ratio_v*like_ratio_w*like_ratio_x *like_ratio_y*like_ratio_gail
  print('---------------- nat_param ---------------------')
  print('all_nat_param_v')
  print(all_nat_param_v[counter,:])
  print('all_nat_param_w')
  print(all_nat_param_w[counter,:])
  print('all_nat_param_x')
  print(all_nat_param_x[counter,:])
  print('all_nat_param_y')
  print(all_nat_param_y[counter,:])
  counter+=1
 est_naive=np.empty((levels_mult.size,save_iter.size))
 std_naive=np.empty((levels_mult.size,save_iter.size))
 est_ce=np.empty((levels_mult.size,save_iter.size))
 std_ce=np.empty((levels_mult.size,save_iter.size))
 num_events_naive=np.zeros((levels_mult.size,save_iter.size))
 num_events_ce=np.zeros((levels_mult.size,save_iter.size))
 for ii in range(levels_mult.size):
  level=base_mean*levels_mult[ii]
  objs_naive_level=(objs_naive>level).astype(float)
  for jj in range(save_iter.size):
   ns_naive =np.minimum(int((save_iter[jj]+1)*ce_sample_size),sample_size)
   est_naive[ii,jj]=np.mean(objs_naive_level[:ns_naive])
   std_naive[ii,jj]=np.std(objs_naive_level[:ns_naive])
   num_events_naive[ii,jj]=np.sum(objs_naive_level[:ns_naive])
   objs_ce_level=like_ratio[jj,:] *(objs_ce[jj,:]>level).astype(float)
   est_ce[ii,jj]=np.mean(objs_ce_level[:sample_size])
   std_ce[ii,jj]=np.std(objs_ce_level[:sample_size])
   num_events_ce[ii,jj]=np.sum(objs_ce[jj,:]>level)
 filename=os.getcwd()+'/rho='+str(rho)+'_alpha='+str(alpha) +'_cesample='+str(ce_sample_size)+'.h5'
 with h5py.File(filename,'w')as f:
  f.create_dataset('levels_mult',data=levels_mult)
  f.create_dataset('save_iter',data=save_iter)
  f.create_dataset('rho',data=rho)
  f.create_dataset('alpha',data=alpha)
  f.create_dataset('ce_sample_size',data=ce_sample_size)
  f.create_dataset('all_nat_param_v',data=all_nat_param_v)
  f.create_dataset('all_nat_param_w',data=all_nat_param_w)
  f.create_dataset('all_nat_param_x',data=all_nat_param_x)
  f.create_dataset('all_nat_param_y',data=all_nat_param_y)
  f.create_dataset('all_nat_param_gail',data=all_nat_param_gail)
  f.create_dataset('base_v',data=base_v)
  f.create_dataset('base_w',data=base_w)
  f.create_dataset('base_x',data=base_x)
  f.create_dataset('base_y',data=base_y)
  f.create_dataset('base_gail',data=base_gail)
  f.create_dataset('objs_naive',data=objs_naive)
  f.create_dataset('objs_ce',data=objs_ce)
  f.create_dataset('like_ratio_ce',data=like_ratio)
 return est_naive,std_naive,est_ce,std_ce, num_events_naive,num_events_ce
def test_ce_simple(seed_value):
 np.random.seed(seed_value)
 num_agents=2
 base_v=np.abs(np.random.randn(num_agents,2))
 base_w=np.array([[1.8,4.1],[2.1,2.8]])
 base_x=np.column_stack((np.random.randn(num_agents),10*np.random.rand(num_agents)))
 base_y=np.array([[1.2,2.3],[3.1,1.8]])
 print('base_v')
 print(base_v)
 print('base_w')
 print(base_w)
 print('base_x')
 print(base_x)
 print('base_y')
 print(base_y)
 dim_gail=20
 mu=5.*np.random.randn(dim_gail)
 sigma=np.diag(np.random.rand(dim_gail))
 base_gail=np.vstack((mu,sigma))
 print('gail mean')
 print(mu)
 lanes=np.array([4,5])
 ic=init_conds.init_conds(base_v,base_w,base_x,base_y,base_gail,lanes)
 base_mean=np.abs(base_x[0,0])
 levels_mult=np.array([2.,10.,14.,18.])
 print('levels')
 print(levels_mult*base_mean)
 sample_size=int(1000000.)
 num_iter=100
 save_iter=(np.array([30,70,100])-1).astype(int)
 rho=.9
 alpha=.9
 ce_sample_size=10000
 ns_naive=sample_size+int(ce_sample_size*(save_iter[-1]+1))
 obs_v=ic.sample_obs(ic.model_v,ic.nat_base_v,ns_naive)
 obs_w=ic.sample_obs(ic.model_w,ic.nat_base_w,ns_naive)
 obs_x=ic.sample_obs(ic.model_x,ic.nat_base_x,ns_naive)
 obs_y=ic.sample_obs(ic.model_y,ic.nat_base_y,ns_naive)
 obs_gail=ic.sample_obs(ic.model_gail,ic.nat_base_gail,ns_naive,if_gail=True)
 objs_naive=test_objective_simple(obs_v,obs_w,obs_x,obs_y,obs_gail,0.)
 objs_ce=np.empty((save_iter.size,sample_size))
 like_ratio=np.empty((save_iter.size,sample_size))
 sample_sizes=np.full(num_iter,ce_sample_size).astype(int)
 step_sizes=np.full(num_iter,alpha)
 all_nat_param_v,all_nat_param_w,all_nat_param_x, all_nat_param_y,all_nat_param_gail =\
  cross_entropy(ic,rho,levels_mult[-1]*base_mean,num_iter,\
    save_iter,sample_sizes,step_sizes,test_objective_simple)
 print('full likelihoods')
 counter=0
 for num_iter in save_iter:
  print('----------------------- Processing '+str(num_iter+1) +' -------------------')
  obs_v=ic.sample_obs(ic.model_v,all_nat_param_v[counter,:],sample_size)
  obs_w=ic.sample_obs(ic.model_w,all_nat_param_w[counter,:],sample_size)
  obs_x=ic.sample_obs(ic.model_x,all_nat_param_x[counter,:],sample_size)
  obs_y=ic.sample_obs(ic.model_y,all_nat_param_y[counter,:],sample_size) 
  obs_gail=ic.sample_obs(ic.model_gail,all_nat_param_gail[counter,:],sample_size,if_gail=True)
  objs_ce[counter,:]=test_objective_simple(obs_v,obs_w,obs_x,obs_y,obs_gail,0.)
  print('mean_ce')
  print(np.mean(objs_ce[counter,:]))
  like_ratio_v,like_ratio_w,like_ratio_x,like_ratio_y,like_ratio_gail =ic.compute_like_ratio(obs_v,obs_w,obs_x,obs_y,obs_gail,all_nat_param_v[counter,:],all_nat_param_w[counter,:],all_nat_param_x[counter,:],all_nat_param_y[counter,:],all_nat_param_gail[counter,:])
  like_ratio[counter,:]=like_ratio_v*like_ratio_w*like_ratio_x *like_ratio_y*like_ratio_gail
  print('---------------- nat_param ---------------------')
  print('all_nat_param_v')
  print(all_nat_param_v[counter,:])
  print('all_nat_param_w')
  print(all_nat_param_w[counter,:])
  print('all_nat_param_x')
  print(all_nat_param_x[counter,:])
  print('all_nat_param_y')
  print(all_nat_param_y[counter,:])
  counter+=1
 est_naive=np.empty((levels_mult.size,save_iter.size))
 std_naive=np.empty((levels_mult.size,save_iter.size))
 est_ce=np.empty((levels_mult.size,save_iter.size))
 std_ce=np.empty((levels_mult.size,save_iter.size))
 num_events_naive=np.zeros((levels_mult.size,save_iter.size))
 num_events_ce=np.zeros((levels_mult.size,save_iter.size))
 actual=np.empty(levels_mult.size)
 for ii in range(levels_mult.size):
  level=base_mean*levels_mult[ii]
  objs_naive_level=(objs_naive>level).astype(float)
  actual[ii]=1-stats.norm.cdf((level-base_x[0,0])/np.sqrt(base_x[0,1]))
  for jj in range(save_iter.size):
   ns_naive =np.minimum(int((save_iter[jj]+1)*ce_sample_size),sample_size)
   est_naive[ii,jj]=np.mean(objs_naive_level[:ns_naive])
   std_naive[ii,jj]=np.std(objs_naive_level[:ns_naive])
   num_events_naive[ii,jj]=np.sum(objs_naive_level[:ns_naive])
   objs_ce_level=like_ratio[jj,:] *(objs_ce[jj,:]>level).astype(float)
   est_ce[ii,jj]=np.mean(objs_ce_level[:sample_size])
   std_ce[ii,jj]=np.std(objs_ce_level[:sample_size])
   num_events_ce[ii,jj]=np.sum(objs_ce[jj,:]>level)
 filename=os.getcwd()+'/rho='+str(rho)+'_alpha='+str(alpha) +'_cesample='+str(ce_sample_size)+'.h5'
 with h5py.File(filename,'w')as f:
  f.create_dataset('levels_mult',data=levels_mult)
  f.create_dataset('save_iter',data=save_iter)
  f.create_dataset('rho',data=rho)
  f.create_dataset('alpha',data=alpha)
  f.create_dataset('ce_sample_size',data=ce_sample_size)
  f.create_dataset('all_nat_param_v',data=all_nat_param_v)
  f.create_dataset('all_nat_param_w',data=all_nat_param_w)
  f.create_dataset('all_nat_param_x',data=all_nat_param_x)
  f.create_dataset('all_nat_param_y',data=all_nat_param_y)
  f.create_dataset('all_nat_param_gail',data=all_nat_param_gail)
  f.create_dataset('base_v',data=base_v)
  f.create_dataset('base_w',data=base_w)
  f.create_dataset('base_x',data=base_x)
  f.create_dataset('base_y',data=base_y)
  f.create_dataset('base_gail',data=base_gail)
  f.create_dataset('objs_naive',data=objs_naive)
  f.create_dataset('objs_ce',data=objs_ce)
  f.create_dataset('like_ratio_ce',data=like_ratio)
 return est_naive,std_naive,est_ce,std_ce, num_events_naive,num_events_ce,actual
# Created by pyminifier (https://github.com/liftoff/pyminifier)

def sim_objective(obs_wp, obs_w):
  # approach(3, 17)
  return np.power((np.power((obs_wp - 3),2) + np.power((obs_w - 17), 2)), 0.5)

def ce_test_carla_baseline(seed_value, rho, alpha, scenario_id):
  # import zmq
  import time
  import numpy as np
  sys.path.append('../')
  from scenario_worker import run_scenario_with_params


  program_state = dict()
  if_verbose = False
  TEST = 0
  level = 0.5
  np.random.seed(seed_value)
  base_wp=np.array([[2.0, 2.0]])
  base_w=np.array([[2.0,2.0]])
  print('base_wp')
  print(base_wp)
  print('base_w')
  print(base_w)
  
  ic=init_conds.init_conds_test(base_wp,base_w)

  nat_param_wp = ic.nat_base_wp
  nat_param_w = ic.nat_base_w

  best_nat_param_wp = nat_param_wp
  best_nat_paran_w = nat_param_w
  
  sample_size=100
  num_iter=1
  save_iter=(np.array([30,70,100])-1).astype(int)
  

  state_checkpoint_root = './cem/checkpoint/' + scenario_id
  if not os.path.exists(state_checkpoint_root):
    os.mkdir(state_checkpoint_root)
  

  best_obj_mean = float('inf')
  obj_mean = list()
  obj = np.zeros((1, sample_size))
  ############################ cem loop ############################
  for i in range(num_iter):
    program_state['nat_param_wp'] = nat_param_wp
    program_state['nat_param_w'] = nat_param_w
    program_state['best_nat_param_wp'] = best_nat_param_wp
    program_state['best_nat_paran_w'] = best_nat_paran_w
    program_state['iteration']  = i # start from 0
    program_state['best_obj_mean'] = best_obj_mean
    program_state['obj_mean'] = obj_mean
    program_state['stage'] = 1
    

    np.save(state_checkpoint_root+'/cem_iter=%d.npy' % (i), program_state)

    # sample parameters in batch
    obs_wp=ic.sample_obs(ic.model_wp,nat_param_wp,sample_size)
    obs_w=ic.sample_obs(ic.model_w,nat_param_w,sample_size)
    sample_wp = (obs_wp * 10).astype(int)
    sample_w = (obs_w * 120.0) - 60.0

    for j in range(sample_size):
      params = []
      params.append(sample_wp[0][j])
      params.append(sample_w[0][j])
      result = run_scenario_with_params(params, 2000)
      result = float(result.split(',')[1])
      obj[0][j] = result
    
      print(f"iter: {j}/{sample_size}")
      print("=======================================")    
    # for j in range(sample_size):
    #   param1 = sample_wp[0][j]
    #   param2 = sample_w[0][j]
      
    #   start_time = time.time()
    #   print("Sending request iteration %d …" % j)
    #   print(f'param {param1}, {param2}')
    #   # send a task to a worker
    #   socket.send_string(("%d,%d,%d" % (param1, param2, j)))

    #   #  Get the reply.
    #   message = socket.recv()
    #   elapse_time = time.time() - start_time
    #   print("Received reply [ %s ]" % (message))
    #   encoding = 'utf-8'
    #   message = message.decode(encoding)
    #   TTC_str = message.split(',')[1]
    #   obj[0][j] = float(TTC_str)
      
    #   print('elapse_time:', elapse_time)



    program_state['obj'] = obj
    program_state['stage'] = 2
    np.save(state_checkpoint_root+'/cem_iter=%d.npy' % (i), program_state)
    
      

    if TEST:
      obs_wp=ic.sample_obs(ic.model_wp,nat_param_wp, sample_size)
      obs_w=ic.sample_obs(ic.model_w,nat_param_w, sample_size)
      sim_wp = (obs_wp*10).astype(int)
      sim_w = (obs_w*20).astype(int)

    # print(obs_wp)
    # print(obs_w)

    # for j in range(sample_size):
      # run scenario and get f(x), here TTC

    if TEST:
      obj = sim_objective(sim_wp, sim_w)
      # print(obj.shape)
      # exit()
    print('objective_mean', np.mean(obj))
    obj_mean.append(np.mean(obj))
    # break
    # print(obj.shape)
    # find the max parameter
    thresh_ind= int(np.floor(rho*sample_size))
    inds=np.argsort(obj,axis=-1)
    print(thresh_ind, inds)
    if(obj[0, inds[0, thresh_ind]] < level):
      print(f'reach the level: at iteration {i} with rho{rho}')
      print('obj:', obj)

    good_ind = inds[:, 0:thresh_ind]
    print(good_ind)
    print(obj[0, good_ind])
    cur_obj_mean = np.mean(obj[:, good_ind])
    if cur_obj_mean < best_obj_mean:
      best_nat_param_wp = nat_param_wp
      best_nat_param_w = nat_param_w
      best_obj_mean = cur_obj_mean
    
    log_like_ratio_wp, log_like_ratio_w, suff_wp, suff_w,\
      grad_log_partition_wp, grad_log_partition_w =\
        ic.compute_log_like_ratio(obs_wp, obs_w, nat_param_wp,\
           nat_param_w, True)
    log_like_ratio_wp = log_like_ratio_wp[good_ind]
    log_like_ratio_w = log_like_ratio_w[good_ind]
    effective_sample_size = int(sample_size*(rho))
    trunc_suff_wp = suff_wp[:,:,good_ind]
    trunc_suff_w = suff_w[:,:, good_ind]
    imp_surrogate_wp =compute_imp_surrogate(ic,trunc_suff_wp,\
      log_like_ratio_wp,effective_sample_size)
    imp_surrogate_w =compute_imp_surrogate(ic,trunc_suff_w,\
      log_like_ratio_w,effective_sample_size)
    # print(suff_wp)
    # print(suff_wp.shape)
    # print(imp_surrogate_w)
    
    coeff_wp = alpha*imp_surrogate_wp +(1-alpha)*grad_log_partition_wp
    coeff_w = alpha*imp_surrogate_w +(1-alpha)*grad_log_partition_w
    # print(coeff_w)

    # update dist.
    nat_param_wp =ic.optimize_cross_entropy(ic.model_wp,coeff_wp,\
      ic.nat_base_wp,if_verbose=if_verbose,center=ic.nat_base_wp)
    nat_param_w =ic.optimize_cross_entropy(ic.model_w,coeff_w,\
      ic.nat_base_w,if_verbose=if_verbose,center=ic.nat_base_w)
  ############################ end cem loop ############################


  # print('best_obj_mean', best_obj_mean)
  # print(obj_mean)
  # print(type(obj_mean[0]))
  # print(np.min(obj_mean))
  return obj_mean

def ce_test_carla(seed_value, rho, alpha):
  import zmq
  import time
  import numpy as np

  
  #  Socket to talk to server
  context = zmq.Context()
  print("Connecting to worker server…")
  socket = context.socket(zmq.REQ)
  socket.connect("tcp://localhost:5555")
  # socket.connect("tcp://140.113.240.44:5555")


  program_state = dict()
  if_verbose = False
  TEST = 0
  level = 0.5
  np.random.seed(seed_value)
  base_wp=np.array([[2.0, 2.0]])
  base_w=np.array([[2.0,2.0]])
  print('base_wp')
  print(base_wp)
  print('base_w')
  print(base_w)
  
  ic=init_conds.init_conds_test(base_wp,base_w)

  nat_param_wp = ic.nat_base_wp
  nat_param_w = ic.nat_base_w

  best_nat_param_wp = nat_param_wp
  best_nat_paran_w = nat_param_w
  
  sample_size=20
  num_iter=100
  save_iter=(np.array([30,70,100])-1).astype(int)
  # rho=.3
  # alpha=.6
  

  best_obj_mean = float('inf')
  obj_mean = list()
  obj = np.zeros((1, sample_size))
  ############################ cem loop ############################
  for i in range(num_iter):
    program_state['nat_param_wp'] = nat_param_wp
    program_state['nat_param_w'] = nat_param_w
    program_state['best_nat_param_wp'] = best_nat_param_wp
    program_state['best_nat_paran_w'] = best_nat_paran_w
    program_state['iteration']  = i # start from 0
    program_state['best_obj_mean'] = best_obj_mean
    program_state['obj_mean'] = obj_mean
    program_state['stage'] = 1
    

    np.save('checkpoint/3_t1-2_0_m_f_l_1_0/cem_iter=%d.npy' % (i), program_state)

    # sample parameters in batch
    obs_wp=ic.sample_obs(ic.model_wp,nat_param_wp,sample_size)
    obs_w=ic.sample_obs(ic.model_w,nat_param_w,sample_size)
    sample_wp = (obs_wp * 10).astype(int)
    sample_w = (obs_w * 120.0) - 60.0
    
    for j in range(sample_size):
      param1 = sample_wp[0][j]
      param2 = sample_w[0][j]
      
      start_time = time.time()
      print("Sending request iteration %d …" % j)
      print(f'param {param1}, {param2}')
      # send a task to a worker
      socket.send_string(("%d,%d,%d" % (param1, param2, j)))

      #  Get the reply.
      message = socket.recv()
      elapse_time = time.time() - start_time
      print("Received reply [ %s ]" % (message))
      encoding = 'utf-8'
      message = message.decode(encoding)
      TTC_str = message.split(',')[1]
      obj[0][j] = float(TTC_str)
      
      print('elapse_time:', elapse_time)
    program_state['obj'] = obj
    program_state['stage'] = 2
    np.save('checkpoint/3_t1-2_0_m_f_l_1_0/cem_iter=%d.npy' % (i), program_state)
    
      

    if TEST:
      obs_wp=ic.sample_obs(ic.model_wp,nat_param_wp, sample_size)
      obs_w=ic.sample_obs(ic.model_w,nat_param_w, sample_size)
      sim_wp = (obs_wp*10).astype(int)
      sim_w = (obs_w*20).astype(int)

    # print(obs_wp)
    # print(obs_w)

    # for j in range(sample_size):
      # run scenario and get f(x), here TTC

    if TEST:
      obj = sim_objective(sim_wp, sim_w)
      # print(obj.shape)
      # exit()
    print('objective_mean', np.mean(obj))
    obj_mean.append(np.mean(obj))
    # break
    # print(obj.shape)
    # find the max parameter
    thresh_ind= int(np.floor(rho*sample_size))
    inds=np.argsort(obj,axis=-1)
    print(thresh_ind, inds)
    if(obj[0, inds[0, thresh_ind]] < level):
      print(f'reach the level: at iteration {i} with rho{rho}')
      print('obj:', obj)

    good_ind = inds[:, 0:thresh_ind]
    print(good_ind)
    print(obj[0, good_ind])
    cur_obj_mean = np.mean(obj[:, good_ind])
    if cur_obj_mean < best_obj_mean:
      best_nat_param_wp = nat_param_wp
      best_nat_param_w = nat_param_w
      best_obj_mean = cur_obj_mean
    
    log_like_ratio_wp, log_like_ratio_w, suff_wp, suff_w,\
      grad_log_partition_wp, grad_log_partition_w =\
        ic.compute_log_like_ratio(obs_wp, obs_w, nat_param_wp,\
           nat_param_w, True)
    log_like_ratio_wp = log_like_ratio_wp[good_ind]
    log_like_ratio_w = log_like_ratio_w[good_ind]
    effective_sample_size = int(sample_size*(rho))
    trunc_suff_wp = suff_wp[:,:,good_ind]
    trunc_suff_w = suff_w[:,:, good_ind]
    imp_surrogate_wp =compute_imp_surrogate(ic,trunc_suff_wp,\
      log_like_ratio_wp,effective_sample_size)
    imp_surrogate_w =compute_imp_surrogate(ic,trunc_suff_w,\
      log_like_ratio_w,effective_sample_size)
    # print(suff_wp)
    # print(suff_wp.shape)
    # print(imp_surrogate_w)
    
    coeff_wp = alpha*imp_surrogate_wp +(1-alpha)*grad_log_partition_wp
    coeff_w = alpha*imp_surrogate_w +(1-alpha)*grad_log_partition_w
    # print(coeff_w)

    # update dist.
    nat_param_wp =ic.optimize_cross_entropy(ic.model_wp,coeff_wp,\
      ic.nat_base_wp,if_verbose=if_verbose,center=ic.nat_base_wp)
    nat_param_w =ic.optimize_cross_entropy(ic.model_w,coeff_w,\
      ic.nat_base_w,if_verbose=if_verbose,center=ic.nat_base_w)
  ############################ end cem loop ############################


  # print('best_obj_mean', best_obj_mean)
  # print(obj_mean)
  # print(type(obj_mean[0]))
  # print(np.min(obj_mean))
  return obj_mean

def ce_test_carla_para(seed_value, rho, alpha):
  import zmq
  import time
  import os
  import json
  import numpy as np
  
  #  Socket to talk to server
  context = zmq.Context()
  # print("Connecting to worker server…")
  # socket = context.socket(zmq.REQ)
  # socket.connect("tcp://localhost:5555")
  # socket.connect("tcp://140.113.240.44:5555")

  # Socket to send messages on
  sender = context.socket(zmq.PUSH)
  sender.bind("tcp://*:5557")

  # # Socket with direct access to the sink: used to synchronize start of batch
  # sink = context.socket(zmq.PUSH)
  # sink.connect("tcp://localhost:5558")

  program_state = dict()
  if_verbose = False
  TEST = 0
  level = 0.5
  np.random.seed(seed_value)
  base_wp=np.array([[2.0, 2.0]])
  base_w=np.array([[2.0,2.0]])
  print('base_wp')
  print(base_wp)
  print('base_w')
  print(base_w)
  
  ic=init_conds.init_conds_test(base_wp,base_w)

  nat_param_wp = ic.nat_base_wp
  nat_param_w = ic.nat_base_w

  best_nat_param_wp = nat_param_wp
  best_nat_paran_w = nat_param_w
  
  sample_size=20
  num_iter=100
  save_iter=(np.array([30,70,100])-1).astype(int)
  # rho=.3
  # alpha=.6
  

  best_obj_mean = float('inf')
  obj_mean = list()
  obj = np.zeros((1, sample_size))

  iteration_data = dict()
  ############################ cem loop ############################
  for i in range(num_iter):
    program_state['nat_param_wp'] = nat_param_wp
    program_state['nat_param_w'] = nat_param_w
    program_state['best_nat_param_wp'] = best_nat_param_wp
    program_state['best_nat_paran_w'] = best_nat_paran_w
    program_state['iteration']  = i # start from 0
    program_state['best_obj_mean'] = best_obj_mean
    program_state['obj_mean'] = obj_mean
    program_state['stage'] = 1
    

    np.save('checkpoint/5_i-1_0_c_l_f_1_0/cem_iter=%d.npy' % (i), program_state)

    # sample parameters in batch
    obs_wp=ic.sample_obs(ic.model_wp,nat_param_wp,sample_size)
    obs_w=ic.sample_obs(ic.model_w,nat_param_w,sample_size)
    sample_wp = (obs_wp * 10).astype(int)
    sample_w = (obs_w * 120.0) - 60.0

    # init. for work_list
    work_list = list()
    with open('work_result.json', 'w') as file:
      json.dump({}, file, indent=4, sort_keys=True)
      file.close()
    
    sim_start_time = time.time()
    for j in range(sample_size):
      param1 = sample_wp[0][j]
      param2 = sample_w[0][j]
      
      
      print("Sending request iteration %d …" % j)
      print(f'param {param1}, {param2}')

      # send a task to a worker
      sender.send_string(("%d,%f,%d" % (param1, param2, j)))
      start_time = time.time()
      work_data = {'param1': param1, 'param2': param2, 'work_num': j, 'start_time': start_time}
      work_list.append(work_data)


    
    while len(work_list) > 0:
      with open('work_result.json', 'r') as f:
        # work_result = json.load(f) # a dict
        work_result = json.loads(f.read()) # read
        f.close()
        finish_work_nums = work_result.keys()
        print('work_result.keys()', len(work_result.keys()))
        finish_work_nums = [int(e) for e in finish_work_nums]
        for k in reversed(range(len(work_list))):
          work_data = work_list[k]
          now_time = time.time()
          if now_time - work_data['start_time'] > 300.0: # send work again
            print('work ', work_data['work_num'], 'over 60s, resend work')
            param1 = sample_wp[0][work_data['work_num']]
            param2 = sample_w[0][work_data['work_num']]
            # sender = context.socket(zmq.PUSH)
            # sender.bind("tcp://*:5557")
            sender.send_string(("%d,%d,%d" % (param1, param2, work_data['work_num'])))
            work_list[k]['start_time'] = time.time()
            continue
          if work_data['work_num'] in finish_work_nums:
            obj[0][work_data['work_num']] = work_result[str(work_data['work_num'])]['TTC']
            work_list.pop(k)
        print('len work_list', len(work_list))
      time.sleep(3)
    

    work_detail = work_result.values()
    collided_cnt = 0
    for detail in work_detail:
      if detail['collided']:
        collided_cnt += 1

    cr = float(collided_cnt) / len(work_detail)
    sim_period = time.time() - sim_start_time
    print('iter', i)
    print('collision rate', cr)
    print('sample_size:', sample_size)
    print('sim_time: ', sim_period)
    iteration_data[i] = {'collision rate': cr, 'sample_size': sample_size, 'sim_time': sim_period}
    with open('iteration_data.json', 'w') as file:
      json.dump(iteration_data, file, indent=4, sort_keys=True)
      file.close()
      
    program_state['obj'] = obj
    program_state['stage'] = 2
    np.save('checkpoint/5_i-1_0_c_l_f_1_0/cem_iter=%d.npy' % (i), program_state)

    # if cr >= 0.5:
    #   return obj_mean
    
      

    if TEST:
      obs_wp=ic.sample_obs(ic.model_wp,nat_param_wp, sample_size)
      obs_w=ic.sample_obs(ic.model_w,nat_param_w, sample_size)
      sim_wp = (obs_wp*10).astype(int)
      sim_w = (obs_w*20).astype(int)

    # print(obs_wp)
    # print(obs_w)

    # for j in range(sample_size):
      # run scenario and get f(x), here TTC

    if TEST:
      obj = sim_objective(sim_wp, sim_w)
      # print(obj.shape)
      # exit()
    print('objective_mean', np.mean(obj))
    obj_mean.append(np.mean(obj))
    # break
    # print(obj.shape)
    # find the max parameter
    thresh_ind= int(np.floor(rho*sample_size))
    inds=np.argsort(obj,axis=-1)
    print(thresh_ind, inds)
    if(obj[0, inds[0, thresh_ind]] < level):
      print(f'reach the level: at iteration {i} with rho{rho}')
      print('obj:', obj)

    good_ind = inds[:, 0:thresh_ind]
    print(good_ind)
    print(obj[0, good_ind])
    cur_obj_mean = np.mean(obj[:, good_ind])
    if cur_obj_mean < best_obj_mean:
      best_nat_param_wp = nat_param_wp
      best_nat_param_w = nat_param_w
      best_obj_mean = cur_obj_mean
    
    log_like_ratio_wp, log_like_ratio_w, suff_wp, suff_w,\
      grad_log_partition_wp, grad_log_partition_w =\
        ic.compute_log_like_ratio(obs_wp, obs_w, nat_param_wp,\
           nat_param_w, True)
    log_like_ratio_wp = log_like_ratio_wp[good_ind]
    log_like_ratio_w = log_like_ratio_w[good_ind]
    effective_sample_size = int(sample_size*(rho))
    trunc_suff_wp = suff_wp[:,:,good_ind]
    trunc_suff_w = suff_w[:,:, good_ind]
    imp_surrogate_wp =compute_imp_surrogate(ic,trunc_suff_wp,\
      log_like_ratio_wp,effective_sample_size)
    imp_surrogate_w =compute_imp_surrogate(ic,trunc_suff_w,\
      log_like_ratio_w,effective_sample_size)
    # print(suff_wp)
    # print(suff_wp.shape)
    # print(imp_surrogate_w)
    
    coeff_wp = alpha*imp_surrogate_wp +(1-alpha)*grad_log_partition_wp
    coeff_w = alpha*imp_surrogate_w +(1-alpha)*grad_log_partition_w
    # print(coeff_w)

    # update dist.
    nat_param_wp =ic.optimize_cross_entropy(ic.model_wp,coeff_wp,\
      ic.nat_base_wp,if_verbose=if_verbose,center=ic.nat_base_wp)
    nat_param_w =ic.optimize_cross_entropy(ic.model_w,coeff_w,\
      ic.nat_base_w,if_verbose=if_verbose,center=ic.nat_base_w)
  ############################ end cem loop ############################


  # print('best_obj_mean', best_obj_mean)
  # print(obj_mean)
  # print(type(obj_mean[0]))
  # print(np.min(obj_mean))
  return obj_mean

def random_sample(seed_value):
  if_verbose = False
  TEST = True
  level = 1
  np.random.seed(seed_value)
  base_wp=np.array([[2.0, 2.0]])
  base_w=np.array([[2.0,2.0]])
  print('base_wp')
  print(base_wp)
  print('base_w')
  print(base_w)
  
  ic=init_conds.init_conds_test(base_wp,base_w)

  nat_param_wp = ic.nat_base_wp
  nat_param_w = ic.nat_base_w

  best_nat_param_wp = nat_param_wp
  best_nat_paran_w = nat_param_w
  
  sample_size=10000
  num_iter=100
  save_iter=(np.array([30,70,100])-1).astype(int)
  # rho=.3
  # alpha=.6
  

  best_obj_mean = float('inf')
  obj_mean = list()
  ############################ cem loop ############################
  for i in range(num_iter):
    # sample parameters in batch
    # obs_wp=(ic.sample_obs(ic.model_wp,ic.nat_base_wp,sample_size) * 10).astype(int)
    # obs_w=ic.sample_obs(ic.model_w,ic.nat_base_w,sample_size) * 120 - 60

    if TEST:
      obs_wp=ic.sample_obs(ic.model_wp,nat_param_wp, sample_size)
      obs_w=ic.sample_obs(ic.model_w,nat_param_w, sample_size)
      sim_wp = (obs_wp*10.0).astype(int)
      sim_w = (obs_w*120-60.0).astype(int)

    print(obs_wp)
    print(obs_w)

    # for j in range(sample_size):
      # run scenario and get f(x), here TTC

    if TEST:
      obj = sim_objective(sim_wp, sim_w)
      print('objective_mean', np.mean(obj))
    obj_mean.append(np.mean(obj))

  return obj_mean

if __name__ == '__main__':
  start = time.time()
  cem_obj = ce_test_carla_baseline(12345, 0.2, 0.8, '5_i-1_0_c_l_f_1_0')
  print('total cem time:', time.time() - start)
  exit()
  # start = time.time()
  # cem_obj = ce_test_carla_para(12345, 0.2, 0.8)
  # print('total cem time:', time.time() - start)
  # exit()
  # mean_dis = []
  # for i in range(1, 2):
  #   mean_dis.append(ce_test_carla(12345, i*0.1, 0.8))
  # # np.save('rho_sweep_0.1to0.9_alpha0.8.npy', mean_dis, True)
  cem_min_list = list()
  cem_mean_list = list()
  random_min_list = list()
  random_mean_list = list()

  import random
  # print(len(mean_dis[0]))
  for i in range(10):
    seed = random.randint(0,1000000)
    cem_obj = ce_test_carla(seed, 0.2, 0.8)
    cem_min = np.min(cem_obj)
    cem_mean = np.mean(cem_obj)
    random_obj = random_sample(seed)
    random_min = np.min(random_obj)
    random_mean = np.mean(random_obj)

    print('cem_obj')
    print('  min:', cem_min)
    print('  average', cem_mean)
    print('random_obj')
    print('  min:', random_min)
    print('  average', random_mean)
    cem_min_list.append(cem_min)
    cem_mean_list.append(cem_mean)
    random_min_list.append(random_min)
    random_mean_list.append(random_mean)

    print('cem_obj')
    print('  min:', cem_min_list)
    print('  average', cem_mean_list)
    print('random_obj')
    print('  min:', random_min_list)
    print('  average', random_mean_list)

  # mean_dis = np.load('rho_sweep_0.1to0.9_alpha0.8.npy')
  
  # import matplotlib.pyplot as plt
  # plt.clf()
  # X = range(1, 101)
  # for i in range(0, 1):
  #   plt.plot(X, mean_dis[i])

  # # plt.plot(TTC_X, TTC_Y_all)
  # # plt.legend(['rho =' + str(round(a*0.1, 1)) for a in range(1, 10)], loc='best')
  # plt.title("best rho=0.1 alpha=0.8") # title
  # plt.xlabel("iteration") # y label
  # plt.ylabel("sample mean distance") # x label
  # # plt.show()
  # plt.savefig('best_rho0.1_alpha0.8.png')


  # # min of mean_dis
  # plt.clf()
  # X = range(1, 10)
  # X = [a * 0.1 for a in X]

  # min_dis = [np.min(a) for a in mean_dis]
  # plt.plot(X, min_dis)

  # # plt.legend(X, loc='best')
  # plt.title("rho sweep") # title
  # plt.xlabel("rho") # y label
  # plt.ylabel("minimum of mean distance from 100 iterations") # x label
  # # plt.show()
  # plt.savefig('rho sweep min.png')

  


# cem: add cem loop     

# Please run "python ce_test.py" to test the cem loop 
# Explanation:
# ce_test_carla() aims to search parameters space to approach
# the objective function.  

# Paramters:                               
# wp: domain(0, 20), Beta distribution
# w:  domain(0, 10), Beta distribution

# Objective function:
# Return the distance to the point (w=3, wp=17)