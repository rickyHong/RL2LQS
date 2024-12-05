import torch
import time
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from logging import getLogger

from QEnv import QEnv as Env
from QModel import Actor as Model
from Replay_buffer import Memory as Memory


static_step = 0.1

class QEvolutionStrategy:
    def __init__(self,
                 es_params,
                 env_params,
                 model_params,
                 optimizer_params,
                 run_params):
        
        # es para
        self.sample_size = es_params['sample_size']
        self.tree_size = es_params['num_test_paras']
        self.history_size = es_params['history_size']
        self.halting = es_params['halting'] # To prevent algorithm from taking too long trajectory
        self.esmode = es_params['ESmode']
        
        # env & para
        self.env = Env(**env_params)
        self.qubit_size = env_params['qubit_size']
        self.circuit_HEA = env_params['HEA']
        self.circuit_length = env_params['length']
        self.pauli_dim = 3*self.qubit_size*(self.circuit_length + 1)
        
        # ES variables
        self.theta0 = None
        self.epsilon = None
        self.count0 = None
        
        # Model
        self.para_space1 = torch.tensor(model_params['para_space1'])
        self.para_space2 = torch.tensor(model_params['para_space2'])
        
        # Memory
        self.mem = Memory(es_params = es_params,
                          env_params = env_params,
                          model_params = model_params,
                          optimizer_params = optimizer_params,
                          run_params = run_params)
        
        # policy
        self.Policy = Policy(es_params = es_params)
    
    
    def set_pauli_operators(self):
        self.env.set_pauli_operators() # define Pauli bases to calcultate unitary operators
    
    
    def evolve(self, batch_size, target_success, epi_num, result_folder, model, cuda_num):
        
        ###############
        # Ready
        ###############
        self.mem.batch_size = batch_size
        self.mem.reset_buffer()
        self.env.load_problems(batch_size, self.tree_size, target_success) #load true_kets of shape (batch_size,pomo_size,basis)
        self.env.reset() # initialize measurement_cnt, theta, zero_vector, finished, fidelity
        
        
        ###############
        # local memory
        ###############
        loop_cnt = 0
        terminal_loop = torch.zeros((batch_size,1))
        theta_all = torch.empty((batch_size, self.tree_size, self.sample_size+1, self.pauli_dim), dtype=torch.float64)
        success_count_all = -0.5*torch.ones((batch_size, self.tree_size, self.sample_size+1), dtype=torch.float64)
        count_state = -0.5*torch.ones((batch_size, self.history_size), dtype=torch.float64)
        
        success_count_list = -0.5*torch.ones((batch_size, self.halting+self.history_size-1), dtype=torch.float64)
        prob_idx_list = torch.zeros((batch_size, self.tree_size, self.halting), dtype=torch.long)
        success_count_all_list = -0.5*torch.ones((batch_size, self.tree_size, self.sample_size+1, self.halting+self.history_size-1), dtype=torch.float64)
        
        
        if self.esmode == 'inference' or self.esmode == 'inference_max':
            pol_list = -torch.ones((batch_size, self.halting))
        
        
        # #############################
        # Initial measurement
        # #############################
        self.count0 = self.env.measure_until_fail() # measure self.env.theta = self.env.that0
        self.env.fidelity_calculation()
        
        while (self.env.finished == 0).any():
            
            ###############################
            # Make states for network
            ###############################
            self.theta0 = self.env.theta # shape: (batch_size, 1, 1, pauli_dim)
            count_est = self.count0[:,0,0]
            success_count_list[:,loop_cnt + self.history_size-1] = count_est
            count_state = success_count_list[:,loop_cnt:loop_cnt+self.history_size]
            
            
            # #################################
            # Policy for hyperparameters & ARS
            # ###############################################################
            # Action Repetition Strategy (ARS) with parameter (t_l,t_u,T_th),
            # where t_l = min_len, t_u=max_len, and T_th = max_epi
            # t_repetition = block_size
            # See equation (11) in the paper
            #################################################################
            min_len = 300
            max_len = 2000
            max_epi = 500
            block_size = max(np.ceil(((min_len - max_len)/max_epi)*epi_num + max_len),min_len)
            
            
            if loop_cnt%block_size ==0:
                
                # static hyperparas
                if self.esmode == 'static':
                    test_points, sample_idx, para_idx, _ = self.Policy.static(loop_cnt, batch_size, self.tree_size)
                

                # Multinomial
                if self.esmode =='train' or self.esmode =='inference':
                    test_points, sample_idx, para_idx, probability = self.Policy.gen_paras(loop_cnt, batch_size, self.mem, model,
                                                                                           count_state, success_count_all, 
                                                                                           self.para_space1, self.para_space2, 
                                                                                           self.tree_size)

                # Greedy
                if self.esmode =='inference_max':
                    test_points, sample_idx, para_idx, probability = self.Policy.gen_paras_inference_max(loop_cnt, batch_size, self.mem, model,
                                                                                                         count_state, success_count_all, 
                                                                                                         self.para_space1, self.para_space2, 
                                                                                                         self.tree_size)
            
            
            
            
            # #############################
            # Sampling and measurement
            # #############################
            theta_sample = self.samples_around_theta(batch_size, test_points) # shape: (batch_size, tree_size, sample_size, pauli_dim)
            self.env.theta = theta_sample
            success_count_sample = self.env.measure_until_fail() # current self.env.theta = theta_samples # shape: (batch_size, tree_size, sample_size)
            sample_fidelity = success_count_sample/(1+success_count_sample)
            success_count_sample_sum = success_count_sample.float().sum(2)
            
            
            ##############################################
            # Combine resutls from theta0 and theta_sample
            ##############################################
            success_count_all[:,:,0:1] = self.count0
            success_count_all[:,:,1:] = success_count_sample
            theta_all[:,:,0:1,:] = self.theta0.expand(-1,self.tree_size,-1,-1)
            theta_all[:,:,1:,:] = theta_sample
            # success_count_all shape: (batch_size, tree_size, sample_size + 1)
            # theta_all shape: (batch_size, tree_size, sample_size + 1, pauli_dim)
            
            
            # #######################################################################
            # Select a case according to the policy given by para_idx (batch_size, 1)
            # #######################################################################
            para_idx_theta = para_idx[:,:,None,None].expand(-1,-1,self.sample_size+1,self.pauli_dim) # shape: (batch_size,1,sample_size+1,pauli_dim)
            theta_all_policy = torch.gather(theta_all,1,para_idx_theta) # shape: (batch_size, 1, sample_size+1, pauli_dim)
            self.env.theta = theta_all_policy
            self.env.fidelity_calculation() # self.env.fidelity shape: (batch_size, 1, sample_size+1)
            fidelity_theta_all_policy = self.env.fidelity.clone()
            
            success_idx = para_idx[:,:,None].expand(-1,-1,self.sample_size+1)
            success_count_policy = torch.gather(success_count_all,1,success_idx) #shape: (batch_size, 1, sample_size+1) choose one sigma 
            
            
            # ######################################################################
            # Update i) success count, ii) fidelity, iii) estimate, and iv) finished
            # ######################################################################
            self.env.add_count(success_count_policy, self.sample_size)
            terminal_loop = torch.where(self.env.finished==0,terminal_loop + torch.ones_like(terminal_loop),terminal_loop)
            
            
            # #########################################
            # Evolution of theta and measure the theta
            # #########################################
            theta0_next = self.theta_evolution(batch_size, test_points, success_count_all) # shape: (batch_size, tree_size, 1, pauli_dim)
            self.env.theta = theta0_next
            self.env.fidelity_calculation() # measure updated self.env.theta"s" # shape: (batch_size, tree_size, 1)
            fidelity_next_theta0 = self.env.fidelity.clone() # shape: (batch_size, tree_size, 1)
            
            para_idx_next = para_idx[:,:,None,None].expand(-1,-1,-1,self.pauli_dim)
            theta0_next_policy = torch.gather(theta0_next,1,para_idx_next)
            self.env.theta = theta0_next_policy
            self.count0 = self.env.measure_until_fail()
            
            
            # #############################
            # Lists
            # #############################
            prob_idx_list[:,:,loop_cnt] = sample_idx
            success_count_all_list[:,:,:,loop_cnt+self.history_size-1] = success_count_all
            
            
            # ###############################
            # Terminal condition & Next state
            # ###############################
            loop_cnt += 1
            terminal_loop = torch.where(self.env.finished==0, loop_cnt*torch.ones_like(terminal_loop), terminal_loop)
            
            
            if loop_cnt >= self.halting:
                self.env.measurement_cnt = torch.where(self.env.finished==0,self.env.measurement_cnt + 
                                                       torch.ones_like(self.env.measurement_cnt)*10**(5+self.qubit_size),self.env.measurement_cnt)
                break
        
        
        # #############################
        # Store data
        # #############################
        self.mem.save(epi_num, terminal_loop, prob_idx_list, success_count_all_list[:,0,0:1,:])
        
        cnt = self.env.measurement_cnt
        fidelity_estimate = self.env.fidelity_update
        self.env.theta = self.env.estimate
        self.env.unitary = self.env.HEA_unitary.get_unitary_HEA(self.env.theta)
        self.env.state = self.env.unitary[:,:,0,:].conj().squeeze(1)
        
        
        # #############################
        # DATA SHARING
        # #############################
        if self.esmode =='train':
            
            print('process num: ',cuda_num)

            mem_dict = {
                'terminal' : self.mem.terminal_loop_buffer,
                'prob_idx' : self.mem.prob_idx_buffer,
                'cnt' : self.mem.cnt_buffer,
            }
            
            
            return cnt, fidelity_estimate, terminal_loop.flatten(), mem_dict
        
        
        if self.esmode=='inference' or self.esmode=='inference_max':

            print("process num: ", cuda_num)
            
            mem_dict = {
                'policy': 0
            }
            
            return cnt, fidelity_estimate, terminal_loop.flatten(), mem_dict
        
        if self.esmode=='static':
            mem_dict = {
                'static': 0
            }
            
            return cnt, fidelity_estimate, terminal_loop.flatten(), mem_dict
    
    
    
    def theta_evolution(self, batch_size, test_points, success_cnt_all):
        # test_paras shape: (batch_size, para_size, 2)
        
        learning_rate = test_points[:,:,0:1,None].expand(-1,-1,-1,self.pauli_dim) # shape: (batch_size, tree_size, 1, pauli_dim)
        step_size = test_points[:,:,1:,None].expand(-1,-1,self.sample_size,self.pauli_dim) # shape: (batch_size, tree_size, sample_size, pauli_dim)
        
        mean_success_cnt = success_cnt_all[:,:,1:].double().mean(2)+1
        success_cnt_ratio = torch.div(success_cnt_all[:,:,1:],mean_success_cnt[:,:,None]) # shape: (batch_size, tree_size, sample_size)
        variations = torch.div(torch.einsum('bpsk,bps->bpsk',self.epsilon,success_cnt_ratio), step_size) # shape: (batch_size, tree_size, sample_size, pauli_dim)
        es_grad = variations.mean(dim=2)[:,:,None,:] # shape: (batch_size, tree_size, 1, pauli_dim)
        theta0_tree = self.theta0.expand(-1,self.tree_size,-1,-1)
        evolved_theta = theta0_tree + learning_rate*es_grad
        theta0_next = torch.where(self.env.finished[:,:,None,None].expand(-1,-1,-1,self.pauli_dim)==0, torch.fmod(evolved_theta, 4*np.pi), theta0_tree)
        # shape: (batch_size, "tree_size", 1, pauli_dim)
        
        return theta0_next
    
    
    
    def samples_around_theta(self, batch_size, test_paras):
        
        step_size = test_paras[:,:,1:,None].expand(-1,-1,self.sample_size,self.pauli_dim)
        # shape: (batch_size, para_size, sample_size, pauli_dim)
        
        
        # For sampling in ES # , dtype=torch.float64
        mean = torch.zeros((batch_size, self.tree_size, self.sample_size, self.pauli_dim))#, dtype=torch.float64)
        cov = torch.eye(self.pauli_dim)[None,None,None,:,:].expand(batch_size, self.tree_size, self.sample_size, -1, -1)
        distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)
        
        
        # self.env.theta = theta0 + samples, {theta0} U {theta0 + sigma*epsilon}
        lim = 2*np.pi
        self.epsilon = torch.clamp(distrib.sample(), min=-lim, max=lim) # shape: (batch_size, para_size, sample_size, pauli_dim)
        theta_sample = self.theta0.expand(-1,self.tree_size,self.sample_size,-1) + step_size*self.epsilon # shape: (batch_size, para_size, sample_size, pauli_dim)
        theta_sample = torch.fmod(theta_sample,4*np.pi) # shape: (batch_size, para_size, sample_size+1, pauli_dim)
        
        return theta_sample




# ###########################################################
# To generate policy according to mode of evoluation strategy
# ###########################################################
class Policy():
    def __init__(self, es_params):
        
        # es para
        self.sample_size = es_params['sample_size']
        self.tree_size = es_params['num_test_paras']
        self.history_size = es_params['history_size']
        self.halting = es_params['halting'] # To prevent algorithm from taking too long trajectory
        self.esmode = es_params['ESmode']
        
        # static
        self.static_lr = es_params['static_p1'] # learning rate
        self.static_ss = es_params['static_p2'] # step-size
    
    
    def static(self, loop_cnt, batch_size, para_size):
        
        test_para = torch.empty((batch_size, para_size, 2))
        test_para[:,:,0] = torch.ones_like(test_para[:,:,0])*self.static_lr
        test_para[:,:,1] = torch.ones_like(test_para[:,:,1])*self.static_ss
        test_points = 10**test_para
        
        sample_idx = torch.zeros((batch_size,para_size)).long()
        sampled_sample_idx = torch.randint(0,para_size,(batch_size,1))
        
        return test_points, sample_idx, sampled_sample_idx, 0
    
    
    def gen_paras(self, loop_cnt, batch_size, mem, model, cnt_state, success_cnt_all, para_space1, para_space2, para_size):    
        
        space_size1 = len(para_space1)
        space_size2 = len(para_space2)
        space_size = space_size1 * space_size2
        assert para_size <= space_size
        
        para_space1_batch = para_space1[None,:].expand(batch_size,-1)
        para_space2_batch = para_space2[None,:].expand(batch_size,-1)
        test_para = torch.empty((batch_size, para_size, 2))
        test_idx = torch.empty((batch_size, 2))
        
        states = mem.state_generation(cnt_state)
        probability = model.forward(states) # policy's shape: (batch_size, para_space)
        probability = probability.detach() #+ 0.01 # 0.01 is noise for exploration
        
        
        #-------------------------------------------------------------------------------------------------
        #lr = [lr0,lr1,lr2] >> space_size1 = 3
        #ss = [s0,s1] >> space_size2 = 2
        # joint probability = [lr0*s0, lr1*s0, lr2*s0, lr0*s1, lr1*s1, lr2*s1]
        # index = [0,1,2,3,4,5] << lr_index = [0,1,2,0,1,2] & sigma_index = [0,0,0,1,1,1]
        #-------------------------------------------------------------------------------------------------
        sample_idx = torch.multinomial(probability, para_size) # shape: (batch_size,para_size)
        lr_sample_idx = torch.fmod(sample_idx, space_size2)
        sigma_sample_idx = torch.div(sample_idx, space_size1, rounding_mode='floor')
        
        
        test_para[:,:,0] = torch.gather(para_space1_batch, 1, lr_sample_idx) # learning_rate
        test_para[:,:,1] = torch.gather(para_space2_batch, 1, sigma_sample_idx) # step_size
        test_points = 10**test_para
        
        sampled_sample_idx = torch.zeros((batch_size,1)).long()
        
        return test_points, sample_idx, sampled_sample_idx, probability
        #return test_points, sample_idx, sampled_sample_idx, probability
        # test_para shape: (batch_size, para_size, 2)
        # para_dix shape: (batch_size, 1)
        # prob_lr & prob_sigma shape: (batch_size, space_size)
        
    
    
    def gen_paras_inference_max(self, loop_cnt, batch_size, mem, model, cnt_state, success_cnt_all, para_space1, para_space2, para_size):
        
        space_size1 = len(para_space1)
        space_size2 = len(para_space2)
        space_size = space_size1 * space_size2
        assert para_size <= space_size
        
        para_space1_batch = para_space1[None,:].expand(batch_size,-1)
        para_space2_batch = para_space2[None,:].expand(batch_size,-1)
        test_para = torch.empty((batch_size, para_size, 2))
        test_idx = torch.empty((batch_size, 2))
        
        
        states = mem.state_generation(cnt_state)
        probability = model.forward(states) # policy's shape: (batch_size, para_space)
        probability = probability.detach()
        
        sample_idx = torch.argmax(probability, dim=1, keepdim=True).expand(-1,para_size)
        lr_sample_idx = torch.fmod(sample_idx, space_size1)
        sigma_sample_idx = torch.div(sample_idx, space_size2, rounding_mode='floor')
        
        test_para[:,:,0] = torch.gather(para_space1_batch, 1, lr_sample_idx.long()) # learning_rate
        test_para[:,:,1] = torch.gather(para_space2_batch, 1, sigma_sample_idx.long()) # step_size
        test_points = 10**test_para
        
        para_idx = torch.zeros((batch_size, 1)).long() #torch.gather(sample_idx,1,sampled_sample_idx)
        
        
        return test_points, sample_idx, para_idx, probability
        #return test_points, sample_idx, sampled_sample_idx, probability
        # test_para shape: (batch_size, para_size, 2)
        # para_dix shape: (batch_size, 1)
        # prob_lr & prob_sigma shape: (batch_size, space_size)
















