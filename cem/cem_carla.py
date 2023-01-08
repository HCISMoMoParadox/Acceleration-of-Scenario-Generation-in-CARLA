import numpy as np,numpy.matlib
from scipy import special as sp,stats
from scipy.special import logsumexp
import math
import init_conds
import argparse
import copy
import zmq
from joblib import Parallel,delayed
from multiprocessing import Process
import time
from ce_test import test_objective,test_objective_simple,test_gradient,test_ce,test_ce_simple
import h5py,os
import sys

def cross_entropy(ic, rho, level, num_iter, save_iter, sample_sizes, step_sizes,
                  compute_objective,  # source_worker_socket,source_sink_req_socket,\
                  # source_worker_direct_sockets,sockets=0,
                  if_gail_covar=False,\
                  if_verbose=False, if_early_stop=False, if_update_gail=True):
    """cross_entropy(ic, rho, num_iter, save_iter, sample_sizes, step_sizes)
    Runs the cross-entropy method with the following inputs:
    
    ic is the init_conds class containing the base parameters
    
    rho is a number between (0, 1) specifying the cut-off quantile 
    at each iteration
    
    num_iter is a positive integer specifying number of cross entropy updates
    
    sample_sizes is a num_iter dimensional vector of positive integers 
    that represents the number of samples at each iteration
    
    step_sizes is a num_iter dimensional vector with values in [0, 1] that 
    determines the amount of weight on the current surrogate candidate q_k(x) 
    \propto \indic{\obj(x) \ge \gamma_k} p_0(x)
    """
    nat_param_wp = ic.nat_base_wp
    nat_param_w = ic.nat_base_w
    # nat_param_gail = ic.nat_base_gail
    all_nat_param_wp = np.empty((save_iter.size+1,)+nat_param_wp.shape)
    all_nat_param_w = np.empty((save_iter.size+1,)+nat_param_w.shape)
    # all_nat_param_gail = np.empty((save_iter.size+1,)+nat_param_gail.shape)
    print(" ------------------------------------------------------------ ")
    print(" --------------- Starting Cross Entropy Loop----------------- ")
    current_best = -1e5
    save_counter = 0
    for iter in range(0, num_iter):
        print(" ================= Iteration = " +
              str(iter) + " =================== ")
        obs_wp = ic.sample_obs(ic.model_v, nat_param_wp, sample_sizes[iter])
        obs_w = ic.sample_obs(ic.model_w, nat_param_w, sample_sizes[iter])
        # obs_gail = ic.sample_obs(
            # ic.model_gail, nat_param_gail, sample_sizes[iter], if_gail=True)
        # obj=compute_objective(obs_v,obs_w,obs_x,obs_y,obs_gail,ic,iter,source_worker_socket,source_sink_req_socket,if_update_gail=if_update_gail)
        # test
        obj = compute_objective(obs_wp, obs_w)
        # test
        thresh_ind = np.int(np.floor(rho*sample_sizes[iter]))
        inds = np.argpartition(obj, thresh_ind)
        if obj[inds[thresh_ind]] > level:
            if if_early_stop:
                for ss in range(save_counter, save_iter.size):
                    all_nat_param_wp[ss, :] = nat_param_wp
                    all_nat_param_w[ss, :] = nat_param_w
                    # all_nat_param_gail[ss, :] = nat_param_gail
                    print('=========== Cross Entropy Ended Early ===========')
                break
            else:
                print('========== Level Exceeded ===========')
                good_inds = obj > level
        else:
            good_inds = inds[thresh_ind:]
        print('obj mean')
        print(np.mean(obj))
        print('obj median')
        print(np.median(obj))
        print('obj good mean')
        obj_good_mean = np.mean(obj[good_inds])
        if current_best < obj_good_mean:
            current_best = obj_good_mean
            all_nat_param_v[-1, :] = nat_param_v
            all_nat_param_w[-1, :] = nat_param_w
            all_nat_param_x[-1, :] = nat_param_x
            all_nat_param_y[-1, :] = nat_param_y
            all_nat_param_gail[-1, :] = nat_param_gail
            print('best updated at iter = '+str(iter))
        print(obj_good_mean)
        log_like_ratio_v, log_like_ratio_w, log_like_ratio_x,\
            log_like_ratio_y, log_like_ratio_gail, suff_v,\
            suff_w, suff_x, suff_y, suff_gail, grad_log_partition_v,\
            grad_log_partition_w, grad_log_partition_x,\
            grad_log_partition_y, grad_log_partition_gail\
            = ic.compute_log_like_ratio(obs_v, obs_w, obs_x,
                                        obs_y, obs_gail, nat_param_v, nat_param_w, nat_param_x,
                                        nat_param_y, nat_param_gail, True)
        log_like_ratio_v = log_like_ratio_v[good_inds]
        log_like_ratio_w = log_like_ratio_w[good_inds]
        log_like_ratio_x = log_like_ratio_x[good_inds]
        log_like_ratio_y = log_like_ratio_y[good_inds]
        log_like_ratio_gail = log_like_ratio_gail[good_inds]
        effective_sample_size = (sample_sizes[iter]*(1.-rho))
        trunc_suff_v = suff_v[:, :, good_inds]
        trunc_suff_w = suff_w[:, :, good_inds]
        trunc_suff_x = suff_x[:, :, good_inds]
        trunc_suff_y = suff_y[:, :, good_inds]
        trunc_suff_gail = suff_gail[:, :, good_inds]
        imp_surrogate_v = compute_imp_surrogate(
            ic, trunc_suff_v, log_like_ratio_v, effective_sample_size)
        imp_surrogate_w = compute_imp_surrogate(
            ic, trunc_suff_w, log_like_ratio_w, effective_sample_size)
        imp_surrogate_x = compute_imp_surrogate(
            ic, trunc_suff_x, log_like_ratio_x, effective_sample_size)
        imp_surrogate_y = compute_imp_surrogate(
            ic, trunc_suff_y, log_like_ratio_y, effective_sample_size)
        imp_surrogate_gail = compute_imp_surrogate(
            ic, trunc_suff_gail, log_like_ratio_gail, effective_sample_size)
        coeff_v = step_sizes[iter]*imp_surrogate_v + \
            (1-step_sizes[iter])*grad_log_partition_v
        coeff_w = step_sizes[iter]*imp_surrogate_w + \
            (1-step_sizes[iter])*grad_log_partition_w
        coeff_y = step_sizes[iter]*imp_surrogate_y + \
            (1-step_sizes[iter])*grad_log_partition_y
        coeff_x = step_sizes[iter]*imp_surrogate_x + \
            (1-step_sizes[iter])*grad_log_partition_x
        coeff_gail = step_sizes[iter]*imp_surrogate_gail + \
            (1-step_sizes[iter])*grad_log_partition_gail
        nat_param_v = ic.optimize_cross_entropy(
            ic.model_v, coeff_v, ic.nat_base_v, if_verbose=if_verbose, center=ic.nat_base_v)
        nat_param_w = ic.optimize_cross_entropy(
            ic.model_w, coeff_w, ic.nat_base_w, if_verbose=if_verbose, center=ic.nat_base_w)
        nat_param_y = ic.optimize_cross_entropy(
            ic.model_y, coeff_y, ic.nat_base_y, if_verbose=if_verbose, center=ic.nat_base_y)
        nat_param_x = ic.optimize_cross_entropy(
            ic.model_x, coeff_x, ic.nat_base_x, if_verbose=if_verbose, center=ic.nat_base_x)
        if if_update_gail:
            num_try = 0
            while num_try < 2:
                num_try += 1
                try:
                    nat_param_gail = ic.optimize_cross_entropy(
                        ic.model_gail, coeff_gail, ic.nat_base_gail, if_gail=True, if_gail_covar=if_gail_covar, if_verbose=if_verbose, center=ic.nat_base_gail[0, :])
                    break
                except:
                    print("Mosek Barfed")
        print('nat_param_v')
        print(nat_param_v)
        print('nat_param_w')
        print(nat_param_w)
        print('nat_param_x')
        print(nat_param_x)
        print('nat_param_y')
        print(nat_param_y)
        if iter in save_iter:
            all_nat_param_v[save_counter, :] = nat_param_v
            all_nat_param_w[save_counter, :] = nat_param_w
            all_nat_param_x[save_counter, :] = nat_param_x
            all_nat_param_y[save_counter, :] = nat_param_y
            all_nat_param_gail[save_counter, :] = nat_param_gail
            save_counter += 1
    print(" ----------------------------------------------------------- ")
    print(" --------------- Ending Cross Entropy Loop ----------------- ")
    return all_nat_param_v, all_nat_param_w, all_nat_param_x, all_nat_param_y, all_nat_param_gail


def compute_imp_surrogate(ic, trunc_suff, log_like_ratio, effective_sample_size):
    imp_surrogate = np.empty((trunc_suff.shape[0], trunc_suff.shape[1]))
    for a in range(ic.num_agents):
        for b in range(trunc_suff.shape[1]):
            pos_inds = trunc_suff[a, b, :] > 0
            neg_inds = trunc_suff[a, b, :] < 0
            if np.any(pos_inds):
                pos_part = logsumexp(
                    log_like_ratio[pos_inds]+np.log(trunc_suff[a, b, pos_inds])) - np.log(effective_sample_size)
                pos_part = np.exp(pos_part)
            else:
                pos_part = 0
            if np.any(neg_inds):
                neg_part = logsumexp(
                    log_like_ratio[neg_inds]+np.log(-trunc_suff[a, b, neg_inds])) - np.log(effective_sample_size)
                neg_part = np.exp(neg_part)
            else:
                neg_part = 0
            imp_surrogate[a, b] = pos_part-neg_part
    return imp_surrogate
