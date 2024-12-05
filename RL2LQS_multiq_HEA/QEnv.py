import time
from dataclasses import dataclass
import torch
from scipy.stats import unitary_group
import numpy as np
import torch.nn.functional as F



class QEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.qubit_size = env_params['qubit_size']
        self.basis_size = 2**self.qubit_size
        self.para_size = 4**self.qubit_size -1
        
        # Circuit params
        ###################################
        self.circuit_HEA = env_params['HEA']
        self.circuit_length = env_params['length']
        self.para_size = 3*self.qubit_size*(self.circuit_length+1)
        
        
        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.tree_size = None
        self.sample_size = None
        self.true_ket = None
        self.pauli_basis = None
        self.theta = None
        self.theta0 = None
        self.target_success = None
        # shape: (batch, basis), dtype = complex

        # Dynamic
        ####################################
        self.measurement_cnt = None
        # shape: (batch,)
        self.unitary = None
        # shape: (batch, basis)
        self.finished = None
        self.success_cnt_pomomax_idx = None
        self.success_cnt_pomorand_idx = None
        self.HEA_unitary = HardwareEfficientAnsatz(**env_params)
        
        # For Measurement
        ####################################
        self.zero = None
        self.measure = None  # measurement base
        # shape: (batch, basis)
        
        # History
        ####################################
        self.fidelity = None
        self.fidelity_update = None
        self.mask = None
        # shape: (batch, -)

        # states to return
        ####################################
        self.estimate = None
        self.state = None
        
        
    def Unitary_sampling(self,batch_size):
        # ____________________________________________________________________________________________________________________________
        # To sample quantums state equally distributed on its space
        # Random sampling of parameters can result in unwanted concentration induced by feature of the space
        # We referenced codes in the followings:
        # (1) F. Mezzadri, “How to generate random matrices from the classical compact groups", NOTICES of the AMS, 54, 592, (2007).
        # (2) The section of "Haar-random matrices from the QR decomposition" in https://pennylane.ai/qml/demos/tutorial_haar_measure
        # ____________________________________________________________________________________________________________________________
        
        # Step 1
        A = torch.normal(mean=torch.zeros(batch_size,self.basis_size,self.basis_size),std=torch.ones(batch_size,self.basis_size,self.basis_size))
        B = torch.normal(mean=torch.zeros(batch_size,self.basis_size,self.basis_size),std=torch.ones(batch_size,self.basis_size,self.basis_size))
        Z = torch.complex(A,B).cdouble()
        # Step 2
        Q, R = torch.linalg.qr(Z)
        # Step 3
        dR = torch.div(R, torch.abs(R) + 10**(-9) )
        Lambda = torch.diagonal(dR,dim1=-2,dim2=-1)
        l2 = torch.diag_embed(Lambda)
        # Step 4
        result = torch.bmm(Q,l2)
        return result
    
    
    def load_single_problem(self, batch_size, tree_size, target_success):
        self.batch_size = batch_size
        self.tree_size = tree_size
        self.target_success = torch.tensor(target_success).long()
        single_problem = self.Unitary_sampling(1)[:,0]
        self.true_ket = single_problem[None,:,:].expand(batch_size,tree_size-1)
        # shape: (batch_size, pomo_size, basis_size)
    
    
    def load_problems(self, batch_size, tree_size, target_success):
        self.batch_size = batch_size
        self.tree_size = tree_size
        self.target_success = torch.tensor(target_success).long()
        kets = self.Unitary_sampling(batch_size)[:,0]
        self.true_ket = kets[:,None,:]
        # shape: (batch_size, pomo_size, basis_size)
    
    
    def make_unitary_1q(self):
        # _______________________________________________________________________________________________
        # U(theta) = exp(j/2 * theta.Pauli_operators)
        # U(theta) = I*cos(|theta/2|) + isin(|theta/2|)*(n.Pauli_operators), where n = (theta/2)/|theta/2|
        # self.pauli_basis_1q = [I,X,Y,Z] # shape: (4,2,2)
        # self.vec = [cos(|theta/2|), jn_x*sin(|theta/2|), jn_y*sin(|theta/2|), jn_z*sin(|theta/2|)]
        # shape: (batch_size,para_size,sample_size,3+1)
        # _______________________________________________________________________________________________
        para_size = self.theta.size(1)
        sample_size = self.theta.size(2)
        vec = torch.empty((self.batch_size,para_size,sample_size,self.para_size + 1),dtype=torch.complex128)  # shape: (batch_size,1,1,3+1)
        theta = self.theta.cdouble()/2
        theta_abs = torch.norm(theta,dim=3,keepdim=True) # shape: (batch_size,1,1)
        theta_norm = F.normalize(theta, p=2.0, dim=3, eps=1e-12)#(theta)/theta_abs # shape: (batch_size,1,1,3)
        vec[:,:,:,0:1] = torch.cos(theta_abs)
        vec[:,:,:,1:] = 1j*torch.sin(theta_abs)*theta_norm
        self.unitary = torch.einsum('bpsk,klm -> bpslm',vec.cdouble(),self.pauli_basis_1q)

    
    def make_unitary(self):
        # _______________________________________________________________________________________________
        # As descirbed in self.set_pauli_operators(), unitary operators are determined by Pauli operators and parameters
        # U(theta) = exp(j/2 * theta.Puali_operators)
        # Process below is for exponentiation of the operators (matrices)
        # _______________________________________________________________________________________________
        iH = torch.einsum('bpsk,klm -> bpslm',self.theta.cdouble(),self.pauli_basis)*1j
        iD, V = torch.linalg.eig(iH/2)
        exp_iD = torch.exp(iD)
        expiH = V @ torch.diag_embed(exp_iD) @ torch.linalg.inv(V)
        self.unitary = expiH
        # shape: (batch_size, para_size, 1          , basis_size, basis_size) for initial state
        # shape: (batch_size, para_size, sample_size, basis_size, basis_size) for samples
    
    
    def reset(self):
        # _______________________________________________________________________________________________
        # measurement_cnt: Object
        # theta: parameter of unitary operator 
        # estimate: estimation result
        # finished: to indicate which batch is finished
        # _______________________________________________________________________________________________
        self.measurement_cnt = torch.zeros((self.batch_size, 1), dtype=torch.float64)
        self.theta = torch.clamp(torch.randn((self.batch_size, 1, 1, self.para_size)), min=-4*np.pi, max=4*np.pi)
        self.estimate = torch.zeros((self.batch_size, 1, 1, self.para_size))
        self.finished = torch.zeros((self.batch_size, 1, ), dtype=torch.long)
        self.fidelity_update = torch.zeros((self.batch_size, 1))
        
        
    def fidelity_calculation(self):
        # _______________________________________________________________________________________________
        # We fix our measurement basis |0....0>
        # Our estimate quantums state is U(theta_est)|true_state>
        # Overlap between the two vector is fidelity |<0....0|U(theta_est)|true_state>|**2
        # _______________________________________________________________________________________________
        self.unitary = self.HEA_unitary.get_unitary_HEA(self.theta)
        state_to_be_measured = torch.einsum('bpsij,bpj->bpsi',self.unitary,self.true_ket)
        self.fidelity = torch.abs(state_to_be_measured[:,:,:,0])**2 # p0 = |<0|U(theta)|true>|^2
    
    
    def measure_until_fail(self):
        #start = time.time()
        # _______________________________________________________________________________________________
        # Draw measurement and obtain the number of consecutive zeros
        # Measurement assumbed to be stopped if "1" popped up
        # _______________________________________________________________________________________________
        # theta's shape: (batch_size, tree_size, sample_size+1, para_size)
        # probability for outcome 0
        self.unitary = self.HEA_unitary.get_unitary_HEA(self.theta)
        tree_size = self.unitary.size()
        self.true_ket.expand(-1,tree_size[1],-1)
        Uket = torch.einsum('bpsij,bpj->bpsi',self.unitary,self.true_ket)
        
        success_prob = torch.abs(Uket[:,:,:,0])**2 # shape: (batch_size, tree_size, sample_size+1)
        assert((success_prob<torch.tensor(1.0)).all())
        
        # if success count is N, it means  1-P^N <= rand < 1-P^(N+1)  ==>  N  <= log(rand)/log(P) < N+1
        rand = torch.rand_like(success_prob)
        log_p_rand = torch.log(rand) / torch.log(success_prob)
        success_cnt = torch.floor(log_p_rand).long()
        success_cnt = torch.min(success_cnt, self.target_success)
        
        return success_cnt # shape: (batch_size, tree_size, sample_size+1)

    
    def add_count(self,success_cnt, sample_size):
        # _______________________________________________________________________________________________
        # Update measurement_cnt, estimate, fidelity_update, finished
        # self.theta = (theta0, samples)
        # _______________________________________________________________________________________________
        #success_cnt shape: (batch_size, 1, sample_size+1) 
        update_idx = self.finished==0
        update_size = update_idx.sum()
        
        self.mask = torch.arange(0,sample_size+1)[None,None,:].expand(self.batch_size,-1,-1) # shape: (batch_size, 1, sample_size+1)
        max_cnt = torch.max(success_cnt.clone(),dim=2) # value's shape: (batch_size, 1)
        max_idx = max_cnt.indices # (batch_size,1)
        success_cnt_sum = success_cnt.clone().sum(dim=2)
        max_mask = torch.le(self.mask, torch.ones_like(success_cnt)*max_idx[:,:,None])
        sucess_cnt_until_max = success_cnt.clone() * max_mask # problem
        sucess_cnt_sum_until_max = sucess_cnt_until_max.sum(dim=2)
        theta_max = torch.gather(self.theta,2,max_idx[:,None,:,None].expand(-1,-1,-1,self.para_size)) # shape: (batch_size, 1, 1, para_size)
        # self.fidelity shape: (batch_size, sample_size+1)
        fidelity_max = torch.gather(self.fidelity.squeeze(1),1,max_idx) # shape: (batch_size, 1)
        # fidelity_max = torch.gather(self.fidelity[:,0,:],1,max_idx) # shape: (batch_size, 1)
        
        
        # _______________________________________________________________________________________________
        # what I am doing is that..
        # If max value < target_success (e.g. 10000): sum all values
        # If max value == target_success (e.g. 10000): sum until max_idx
        # ->
        # mask = [0,1,2,...,sample+1]
        # cnt = [10,1,5,100,0,10000,...,10]
        # max_idx = [5]
        # max_mask = [True,True,True,True,True,True,False,...,False]
        # cnt * max_maxk = [10,1,5,100,0,10000,0,0,0,0,0,...,0] -> sum(cnt*max_mask) = cnt_sum_until_max
        # _______________________________________________________________________________________________
        self.measurement_cnt[update_idx] = torch.where(max_cnt.values[update_idx]>=self.target_success,
                                                       self.measurement_cnt[update_idx]+sucess_cnt_sum_until_max[update_idx],
                                                       self.measurement_cnt[update_idx]+success_cnt_sum[update_idx]) # measurement_cnt's shape: (batch_size, tree_size,)
        
        self.estimate[update_idx] = torch.where(max_cnt.values[:,:,None,None].expand(-1,-1,-1,self.para_size)[update_idx]>=self.target_success,
                                                theta_max[update_idx],self.theta[:,:,0,:][:,:,None,:][update_idx]) # estimate's shape: (batch_size, 1, 1, para_size)
        
        self.fidelity_update[update_idx] = torch.where(max_cnt.values[update_idx]>=self.target_success,
                                                      fidelity_max[update_idx],self.fidelity[:,0,0][:,None][update_idx]) # fidelity_update's shape: (batch_size, 1)
        
        self.finished[update_idx] = torch.where(max_cnt.values[update_idx]>=self.target_success, torch.ones_like(self.finished)[update_idx], self.finished[update_idx])

        

#################################
### Hardware efficient ansatz ###
#################################
class HardwareEfficientAnsatz():
    def __init__(self, **env_params):
        
        # Const @INIT
        ####################################
        self.env_params = env_params
        self.qubit_size = env_params['qubit_size']
        self.basis_size = 2**self.qubit_size
        self.circuit_length = env_params['length']
        self.basis_size = 2**self.qubit_size
        self.theta_dim = 3*self.qubit_size*(self.circuit_length+1) # << length of theta
        
        self.cnot_2q = torch.tensor([[1.0,0.0,0.0,0.0],
                                     [0.0,1.0,0.0,0.0],
                                     [0.0,0.0,0.0,1.0],
                                     [0.0,0.0,1.0,0.0]],dtype=torch.complex128)
        self.identity = torch.tensor([[1.0,0.0],[0.0,1.0]],dtype=torch.complex128)
        self.cnot_nq = self.nCNOT()
        
    
    def get_unitary_HEA(self, theta):
        # len(theta) = 3*qubit_size*(circuit_length+1)
        
        batch_size = theta.size(0)
        
        HEA = self.nR(theta[:,:,:,0:3*self.qubit_size])
        
        if self.qubit_size != 1:
            cnot_nq_expand = self.cnot_nq[None,None,None,:,:].expand(batch_size,-1,-1,-1,-1)

            for l in range(self.circuit_length):
                lth_theta = theta[:,:,:,3*self.qubit_size*(l+1):3*self.qubit_size*(l+2)]
                lth_nR = self.nR(lth_theta)
                HEA = torch.einsum('bsrij,bsrjk->bsrik',cnot_nq_expand,HEA)
                HEA = torch.einsum('bsrij,bsrjk->bsrik',lth_nR,HEA)
                
        
        return HEA
    
    
    def nR(self, theta):
        # len(theta) = 3*qubit_size
        
        t1q = theta[:,:,:,0:3]
        y = self.U1q(t1q).clone()
        
        for i in range(self.qubit_size-1):
            ith_t1q = theta[:,:,:,3*(i+1):3*(i+2)]
            u = self.U1q(ith_t1q).clone()
            y = kronecker(y,u)
            
        return y
        
    
    def U1q(self,t1q):
        
        batch_size = t1q.size(0)
        sample_size = t1q.size(2)
        u_1q = torch.empty((batch_size,1,sample_size,2,2),dtype=torch.complex128)
        
        u00 =  1  * torch.cos(t1q[:,:,:,0]/2) * torch.exp(-1j * t1q[:,:,:,2]/2)
        u01 = -1j * torch.sin(t1q[:,:,:,0]/2) * torch.exp(-1j * t1q[:,:,:,2]/2) * torch.exp(-1j*t1q[:,:,:,1])
        u10 = -1j * torch.sin(t1q[:,:,:,0]/2) * torch.exp( 1j * t1q[:,:,:,2]/2) * torch.exp( 1j*t1q[:,:,:,1])
        u11 =  1  * torch.cos(t1q[:,:,:,0]/2) * torch.exp( 1j * t1q[:,:,:,2]/2)
        
        u_1q[:,:,:,0,0] = u00
        u_1q[:,:,:,0,1] = u01
        u_1q[:,:,:,1,0] = u10
        u_1q[:,:,:,1,1] = u11
        
        return u_1q
    
    
    def nCNOT(self):
        cnot = None
        for i in range(self.qubit_size-1):
            
            g = torch.tensor([[1.0]])
            for q in range(self.qubit_size-1):
                if q==i:
                    g = torch.kron(g,self.cnot_2q)
                else:
                    g = torch.kron(g,self.identity)
            
            if i==0:
                cnot = g.clone()
            else:
                cnot = g@cnot
        
        return cnot
    
    
#################################
#### Batch Kronecker Product ####
#################################
def kronecker(A, B):
    #assert(A.size(0)==B.size(0))
    #assert(A.size(1)==B.size(1))
    return torch.einsum("lskab,lskcd->lskacbd", A, B).view(A.size(0),A.size(1),A.size(2),A.size(3)*B.size(3),A.size(4)*B.size(4))   