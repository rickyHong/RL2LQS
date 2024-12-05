import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

import numpy as np
from logging import getLogger

from utils.utils import get_result_folder, TimeEstimator


class Memory():
    def __init__(self,
                 es_params,
                 env_params,
                 model_params,
                 optimizer_params,
                 run_params):
        
        # Const @INIT
        ####################################
        self.es_params = es_params
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.run_params = run_params
        
        # Parameters for buffers
        ####################################
        self.total_episodes = self.run_params['episodes']
        self.batch_size = self.run_params['batch_size']
        self.halting = self.es_params['halting']
        self.memory_window = self.run_params['memory_window']
        self.random_sample = self.model_params['random_selection']
        self.target_success = torch.tensor(self.es_params['target_success']).item()
        self.episodes = torch.div(self.total_episodes,self.batch_size).long()
        self.history_size = self.es_params['history_size']
        self.tree_size = self.es_params['num_test_paras']
        self.sample_size = self.es_params['sample_size']
        self.target_success = self.es_params['target_success']
        self.qubit_size = self.env_params['qubit_size']
        self.reward = self.model_params['reward']
        
        # buffers
        ####################################
        self.terminal_loop_buffer = None
        self.fidelity_buffer = None
        self.finished_buffer = None
        self.traj_ratio = None
        self.tree_idx_list = None
        
        # inputs
        ####################################
        self.qubit_size = env_params['qubit_size']
        self.state = None
        self.sample_mean = None
        self.action = None
        self.finished_dix = None
        
        
    
    ########################
    # INITIALIZE BUFFER
    ########################
    def reset_buffer(self):
        self.idx = 0
        self.terminal_loop_buffer = torch.zeros((self.batch_size*self.memory_window,1))
        self.prob_idx_buffer = torch.zeros((self.batch_size*self.memory_window, self.tree_size, self.halting), dtype=torch.long)
        self.cnt_buffer = -0.5*torch.ones((self.batch_size*self.memory_window, 1, self.halting+self.history_size-1), dtype=torch.float64)
    
    
    ############
    # SAVE
    ############
    def save(self, epi_num, terminal_loop, prob_idx_list, success_count_all_list):
        slot = epi_num%self.memory_window
        idx1 = self.batch_size*slot
        idx2 = self.batch_size*(slot+1)
        
        self.terminal_loop_buffer[idx1:idx2,:] = terminal_loop
        self.prob_idx_buffer[idx1:idx2,:,:] = prob_idx_list
        self.cnt_buffer[idx1:idx2,:,:] = success_count_all_list
    
    
    
    ##########################
    # NETWORK INPUT GENERATION
    ##########################
    def state_generation(self, cnt_input):   
        
        state_batch_size = cnt_input.size()[0]
        states = torch.empty((state_batch_size, self.history_size), dtype=torch.float64)
        states[:,0*self.history_size:1*self.history_size] = cnt_input/self.target_success
        
        return states
 
    
    ########################
    # RANDOM SAMPLING
    ########################
    def random_selection(self,mem_window):
        
        assert((self.random_sample <= self.batch_size*mem_window))
        num_sample = self.model_params['random_selection']

        
        # batch selection
        batch_seed = torch.randperm(self.batch_size)[0:num_sample][:,None]
        batch_seed_cnt = batch_seed[:,:,None].expand(-1, -1, self.halting+self.history_size-1)
        batch_seed_prob_idx = batch_seed[:,:,None].expand(-1,self.tree_size, self.halting)

        
        batch_cnt = torch.gather(self.cnt_buffer[0:self.batch_size*mem_window,:,:],0,batch_seed_cnt)
        batch_prob_idx = torch.gather(self.prob_idx_buffer[0:self.batch_size*mem_window,:,:],0,batch_seed_prob_idx)
        batch_terminals = torch.gather(self.terminal_loop_buffer[0:self.batch_size*mem_window,:],0,batch_seed)

        
        # trajectory selection
        random_number = torch.rand(num_sample)[:,None]
        trajectory_seed = (random_number*batch_terminals).trunc().long() # 0 ~ batch_terminal-1
        
        trajectory_seed_cnt = trajectory_seed[:,None,:]#.expand(-1,1,-1)
        trajectory_seed_prob_idx = trajectory_seed[:,None,:].expand(-1,self.tree_size,-1)
        
        
        sampled_count = torch.gather(batch_cnt,2,trajectory_seed_cnt)[:,0,:]
        sampled_prob_idx = torch.gather(batch_prob_idx,2,trajectory_seed_prob_idx)
        done = None
        
        reward = -(batch_terminals -  trajectory_seed_cnt[:,0,:])/self.halting
        
        sampled_next_count = None
        return sampled_count, sampled_next_count, sampled_prob_idx, done, reward
        
    
    
    ############################
    # TO COMBINE DATA FROM GPUS
    ############################
    def collect_dist_data(self, num_process, gpu_num, epi_num, bsize, mem_dict):
        
        slot = epi_num%self.memory_window
        idx1 = bsize*num_process*slot + bsize*gpu_num
        idx2 = bsize*num_process*slot + bsize*(gpu_num+1)
        
        self.terminal_loop_buffer[idx1:idx2,:] = mem_dict['terminal'].clone()
        self.prob_idx_buffer[idx1:idx2,:,:] = mem_dict['prob_idx'].clone()
        self.cnt_buffer[idx1:idx2,:] = mem_dict['cnt'].clone()
        
        return 0
        
        
        
        
