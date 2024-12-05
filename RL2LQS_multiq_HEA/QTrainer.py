import torch
import time
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

from multiprocessing.connection import wait
import torch.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from logging import getLogger

from QEnv import QEnv as Env
from QES import QEvolutionStrategy as QES
from QModel import Actor as Actor
from QModel import Critic as Critic
from Replay_buffer import Memory as Memory

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import StepLR
from utils.utils import get_result_folder, TimeEstimator



class QTrainer:
    def __init__(self,
                 env_params,
                 es_params,
                 model_params,
                 optimizer_params,
                 run_params):
        
        # save arguments
        self.env_params = env_params
        self.es_params = es_params
        self.model_params = model_params 
        self.optimizer_params = optimizer_params
        self.run_params = run_params
        
        # result folder, logger
        self.logger = getLogger()
        self.result_folder = get_result_folder()
        
        # cuda
        self.USE_CUDA = self.run_params['use_cuda']
        if self.USE_CUDA:
            cuda_device_num = 0 
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.DoubleTensor')
        self.device = device
        
        
        # MEMORY
        self.mem = Memory(es_params = es_params,
                          env_params = env_params,
                          model_params = model_params,
                          optimizer_params = optimizer_params,
                          run_params = run_params)
        
        
        # RL model
        self.Actor = Actor(**self.model_params)
        self.Critic = Critic(**self.model_params)
        self.optimizer_Actor = Optimizer(self.Actor.parameters(), **self.optimizer_params['optimizer'])
        self.optimizer_Critic = Optimizer(self.Critic.parameters(), **self.optimizer_params['optimizer'])
        
        self.esmode = es_params['ESmode']
        
        
        # utility
        self.time_estimator = TimeEstimator()
        
        # multi gpu
        if self.esmode == 'train':
            self.mem_dict0 = {
                'terminal' : None,
                'prob_idx' : None,
                'count': None
            }
            
        if self.esmode == 'inference':
            self.mem_dict0 = { 'policy' : None }
        
        if self.esmode == 'static':
            self.mem_dict0 = { 'static' : None }
    
    
    
    def run(self):
        
        mp.set_start_method('spawn')
        if self.USE_CUDA:
            master_cuda_device_num = 0
            master_cuda_device = torch.device('cuda', master_cuda_device_num)
        else:
            master_cuda_device_num = 0
            master_cuda_device = torch.device('cpu', master_cuda_device_num)
        self.time_estimator.reset()
        
        
        # Parameters
        total_episodes = self.run_params['episodes']
        batch_size = self.run_params['batch_size']
        iterations = torch.div(total_episodes,batch_size).long()
        memory_window = self.run_params['memory_window']
        random_sample = self.model_params['random_selection']
        history_size = self.es_params['history_size']
        para_size = self.es_params['num_test_paras']
        halting = self.es_params['halting']
        target_success = self.es_params['target_success']
        
        
        # Results
        cnt_list_ALL = -torch.ones((batch_size, iterations), dtype=torch.long)
        fidelity_list_ALL = -torch.ones((batch_size, iterations), dtype=torch.float64)
        terminal_loop_ALL = -torch.ones((batch_size, iterations), dtype=torch.float64)
        if self.esmode == ('inference' or 'inference_max'):
            policy_list_ALL = -torch.ones((batch_size, halting, iterations), dtype=torch.float64)
        
        
        # model load
        if self.model_params['load'] == True:
            
            mpath = self.model_params['model_path']
            mpath_actor = mpath + "/trained_actor.pt"
            mpath_critic = mpath + "/trained_critic.pt"
            
            self.Actor.load_state_dict(torch.load(mpath_actor))
            self.Actor.eval()
            
            self.Critic.load_state_dict(torch.load(mpath_critic))
            self.Critic.eval()
                    
        # memory reset
        self.mem.reset_buffer()
        
        
        # GPU multiprocessing
        CUDA_DEVICE_NUM = self.run_params['cuda_device_num']
        num_process = len(self.run_params['cuda_device_num'])
        batch_dist = int(batch_size/num_process)
        q_list = [mp.Queue() for i in range(num_process)]
        rq_list = [mp.Queue() for i in range(num_process)]
        
        
        episode = 0
        epi_num = 0
        while episode < total_episodes:
            process_list = []
            save_value = []
            master_cuda_device_num = 0
            if self.USE_CUDA:
                master_cuda_device = torch.device('cuda', master_cuda_device_num)
            else:
                master_cuda_device = torch.device('cpu', master_cuda_device_num)
            
            
            # distribute tasks
            for i in range(num_process-1, -1, -1):
                cuda_num = CUDA_DEVICE_NUM[i]
                torch.cuda.empty_cache()
                process = mp.Process(target=self.distributed_ES, args=(q_list[i], rq_list[i], i, cuda_num, epi_num, batch_dist,
                                                                       self.result_folder, self.Actor, self.es_params,
                                                                       self.env_params, self.model_params,
                                                                       self.optimizer_params,
                                                                       self.run_params))
                process.start()
                process_list.append(process)
                rq_list[i].put(('dist done'))
            
            
            # wait until done
            for i in range(num_process-1, -1, -1):
                rq_list[i].put(('sent'))
                msg = q_list[i].get()
                idx, cnt, fid, terminal, mem_dict = msg[0], msg[1], msg[2], msg[3], msg[4]
                
                id1 = batch_dist*idx
                id2 = batch_dist*(idx+1)
                cnt_list_ALL[id1:id2,epi_num:epi_num+1] = cnt.to(device=master_cuda_device).clone()
                fidelity_list_ALL[id1:id2,epi_num:epi_num+1] = fid.to(device=master_cuda_device).clone()
                terminal_loop_ALL[id1:id2,epi_num] = terminal.to(device=master_cuda_device).clone()
                
                if self.esmode == ('inference' or 'inference_max'):
                    policy_list_ALL[id1:id2, :, epi_num:epi_num+1] = mem_dict['policy'][:,:,None]
                
                if self.esmode == 'train':
                    for key, val in mem_dict.items():
                        self.mem_dict0[key] = mem_dict[key].to(device=master_cuda_device).clone()
                        del key, val
                        
                    _ = self.mem.collect_dist_data(num_process, idx, epi_num, batch_dist, self.mem_dict0)
                
                del msg, idx, cnt, fid, terminal, mem_dict
            
            
            for rq in rq_list:
                rq.put(('close'))
            
            for p in process_list:
                p.join()
            
            count = cnt_list_ALL[:,epi_num:epi_num+1]
            
            episode += batch_size
            
            if self.esmode == 'train':
                traj_ratio = self.learning(epi_num,halting,memory_window,para_size)
            
            epi_num += 1
            
            
            # Model & Data save
            torch.save(cnt_list_ALL, '{}/cnt.pt'.format(self.result_folder))
            torch.save(fidelity_list_ALL, '{}/fid.pt'.format(self.result_folder))
            torch.save(terminal_loop_ALL, '{}/terminal.pt'.format(self.result_folder))
            
            if self.esmode == 'train':
                torch.save(self.Actor.state_dict(), '{}/trained_actor.pt'.format(self.result_folder))
                torch.save(self.Critic.state_dict(), '{}/trained_critic.pt'.format(self.result_folder))
                
            if self.esmode == ('inference' or 'inference_max'):
                torch.save(policy_list_ALL, '{}/policy.pt'.format(self.result_folder))
                
            
            # Logs
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, total_episodes)
            self.logger.info("[target {:.2e}] episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], cnt[{}], traj[{}]".format(
                target_success, episode, total_episodes, elapsed_time_str, remain_time_str,(count+1).float().log10().mean().item(),traj_ratio))
        
        # Save
        print('save')
        torch.save(cnt_list_ALL, '{}/cnt.pt'.format(self.result_folder))
        torch.save(fidelity_list_ALL, '{}/fid.pt'.format(self.result_folder))
        torch.save(terminal_loop_ALL, '{}/terminal.pt'.format(self.result_folder))
        torch.save(self.Actor.state_dict(), '{}/trained_actor.pt'.format(self.result_folder))
        torch.save(self.Critic.state_dict(), '{}/trained_critic.pt'.format(self.result_folder))
        
        
        # Done, PLOT RESULT
        self.logger.info(" *** Test Done *** ")
        self.logger.info("[target {:.2e}], log(cnt) mean  {:.4f}, log(mean infid) {:.4f}".format(
            target_success, (cnt_list_ALL+1).float().log10().mean(), (1- fidelity_list_ALL).float().log10().mean()))

        return cnt_list_ALL, fidelity_list_ALL
    
    
    
    #########################
    # MULTI GPU
    #########################
    @staticmethod
    def distributed_ES(q, rq, id, cuda_num, epi_num, batch_dist,
                       result_folder, model0, es_params, 
                       env_params,model_params,
                       optimizer_params,
                       run_params):
        
        USE_CUDA = run_params['use_cuda']
        master_cuda_device_num = 0
        
        if USE_CUDA:
            cuda_device_num = 0
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
            cuda_device_num = cuda_num
            torch.cuda.set_device(cuda_device_num)
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.DoubleTensor')
            cuda_device_num = cuda_num

        
        model = Actor(**model_params)
        model.load_state_dict(model0.state_dict())
        
        
        rq.get()
        Evolution_strategy = QES(es_params=es_params,
                                 env_params=env_params,
                                 model_params=model_params,
                                 optimizer_params=optimizer_params,
                                 run_params=run_params)
        
        target_success = es_params['target_success']
        cnt, fidelity, terminal_loop, mem_dict = Evolution_strategy.evolve(batch_dist, target_success, epi_num, result_folder, model, cuda_num)
        
        rq.get() 
        q.put((id, cnt, fidelity, terminal_loop, mem_dict))
        
        while True:
            cmd = rq.get()
            if cmd == 'close':
                del cnt, fidelity, terminal_loop, mem_dict
                del Evolution_strategy, model
                del q, rq, id, cuda_num, epi_num, batch_dist, result_folder, model0
                break
    
    
    #############################
    # RANDOM SAMPLING & LEARNING
    #############################
    def learning(self,epi_num,halting,memory_window,tree_size):
        target_success = self.es_params['target_success']
        
        
        update = 0
        traj_ratio = (self.mem.terminal_loop_buffer.mean()/halting).double()
        update_lim = self.model_params['update']*traj_ratio
        while update < update_lim:
            mem_window = min(epi_num+1,memory_window)
            
            
            # Mini Batch
            current_count, _, prob_idx, done, reward = self.mem.random_selection(mem_window)
            current_state = current_count/target_success
            
            Svalue_target = reward
            Svalue = self.Critic(current_state)
            
            lss = (Svalue_target - Svalue)
            loss_Critic = 0.5*lss*lss
            loss_Critic = loss_Critic.mean()
            
            self.optimizer_Critic.zero_grad()
            loss_Critic.backward()
            self.optimizer_Critic.step()
            
            # Actor
            advantage = lss.detach()
            _, _, log_policy_action12 = self.Actor.forward_to_learn(current_state, prob_idx)
            
            loss_Actor = -log_policy_action12*advantage
            loss_Actor = loss_Actor.mean()
            
            self.optimizer_Actor.zero_grad()
            loss_Actor.backward()
            self.optimizer_Actor.step()
            
            update += 1
        return traj_ratio.item()



def state_function(state, target_success):
    return state/target_success
