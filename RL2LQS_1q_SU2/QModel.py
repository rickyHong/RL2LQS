import torch
import numpy as np
from torch import nn
import torch.nn.functional as F



#######################
# Policy function
#######################
class Actor(nn.Module):
    def __init__(self,**model_params):
        super().__init__()
        self.para_space1 = torch.tensor(model_params['para_space1'])
        self.para_space2 = torch.tensor(model_params['para_space2'])
        
        self.model_params = model_params
        self.input_size = self.model_params['history_size']
        self.para_size1 = len(self.para_space1)
        self.para_size2 = len(self.para_space2)
        self.output_size = self.para_size1 * self.para_size2

        self.dim1 = 50
        self.dim2 = 50
        self.dimf = 50
        
        # Actor
        self.X = nn.Linear(self.input_size, self.dim1)
        self.X1 = nn.Linear(self.dim1, self.dim2)
        self.X2 = nn.Linear(self.dim2, self.dimf)
        self.Xf = nn.Linear(self.dimf, self.output_size)        
    
    
    def forward(self, state):
        
        x = F.relu(self.X(state))
        x = F.relu(self.X1(x))
        x = F.relu(self.X2(x))
        x = self.Xf(x)
        log_prob = F.log_softmax(x,dim=1)
        probability = torch.exp(log_prob)
        
        return probability
    
    
    def forward_to_learn(self,state, prob_idx): 
        
        
        x = F.relu(self.X(state))
        x = F.relu(self.X1(x))
        x = F.relu(self.X2(x))
        x = self.Xf(x)
        
        log_prob = F.log_softmax(x,dim=1)
        
        
        action1, action2, log_policy12 = self.draw_action(log_prob, prob_idx)
        
        return action1, action2, log_policy12

    
    def draw_action(self, log_prob, prob_idx):
        
        batch_size = prob_idx.size(0)
        para_space1_batch = self.para_space1[None,:].expand(batch_size,-1)
        para_space2_batch = self.para_space2[None,:].expand(batch_size,-1)
        prob_idx = prob_idx.squeeze(1)
        
        log_policy12 = torch.gather(log_prob,1,prob_idx)
        
        lr_idx = torch.fmod(prob_idx, self.para_size2)
        sigma_idx = torch.div(prob_idx, self.para_size1, rounding_mode='floor')
        
        action1 = torch.gather(para_space1_batch, 1, lr_idx) # learning_rate
        action2 = torch.gather(para_space2_batch, 1, sigma_idx) # step_size
        
        return action1, action2, log_policy12
    


#######################
# State-Value function
#######################
class Critic(nn.Module):
    def __init__(self,**model_params):
        super().__init__()
        self.model_params = model_params
        
        self.input_size = self.model_params['history_size']
        self.dim1 = 100
        self.dim2 = 100
        self.dimf = 100
        self.output_size = 1
        
        # Critic
        self.X = nn.Linear(self.input_size, self.dim1)
        self.X1 = nn.Linear(self.dim1, self.dim2)
        self.X2 = nn.Linear(self.dim2, self.dimf)
        self.Xf = nn.Linear(self.dimf, self.output_size)
        
    
    def forward(self, state):
        
        x = F.relu(self.X(state))
        x = F.relu(self.X1(x))
        x = F.relu(self.X2(x))
        
        State_value = self.Xf(x)
        
        return State_value