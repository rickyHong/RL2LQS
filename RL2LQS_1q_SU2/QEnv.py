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
        self.make_unitary_2q = Universal_2qubit_Gate(**env_params)
        
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
    
    
    def set_pauli_operators(self):
        # _______________________________________________________________________________________________
        # To generate Pauli operators sigma_x, sigma_y, and sigma_z
        # In short, sigmas can be read X, Y, Z
        # X := [[0,1],[1,0]]
        # Y := [[0,-j],[j,0]]
        # Z := [[1,0],[0,1]]
        # In addition to these, identity operator should be defined I := [[1,0],[0,1]]
        # For a qubit system, an arbitrary unitary operator can be written in terms of sigmas -> U(a,b,c) = exp[t*j* (a*X + b*Y + c*Z)/2] <- 3 operators and parameters
        # For two qubits, to represent arbitrary rotations, we need 15 operators: IX, IY, IZ, XI, YI, ZI, XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ and respective parameters
        # For N qubits, 4**N-1 operators and parameters
        # _______________________________________________________________________________________________
        ixyz = torch.tensor([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]], dtype=torch.complex128)
        pauli_kron = ixyz.clone()
        for _ in range(self.qubit_size-1):
            pauli_kron = torch.kron(pauli_kron,ixyz)
        self.pauli_basis = pauli_kron[1:self.para_size+1]
        self.pauli_basis_1q = pauli_kron
        
        
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
        self.true_ket = kets[:,None,:]#.expand(-1,tree_size,-1)
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
        
        if self.qubit_size == 1:
            self.make_unitary_1q() # SU2
        elif self.qubit_size == 2:
            self.unitary = self.make_unitary_2q.get_unitary2q(self.theta) # SU4
        else:
            self.make_unitary()
        
        state_to_be_measured = torch.einsum('bpsij,bpj->bpsi',self.unitary,self.true_ket)
        self.fidelity = torch.abs(state_to_be_measured[:,:,:,0])**2 # p0 = |<0|U(theta)|true>|^2
    
    
    def measure_until_fail(self):
        # _______________________________________________________________________________________________
        # Draw measurement and obtain the number of consecutive zeros
        # Measurement assumbed to be stopped if "1" popped up
        # _______________________________________________________________________________________________
        # theta's shape: (batch_size, tree_size, sample_size+1, para_size)
        # probability for outcome 0

        if self.qubit_size == 1:
            self.make_unitary_1q() # SU2
        elif self.qubit_size == 2:
            self.unitary = self.make_unitary_2q.get_unitary2q(self.theta) # SU4
        else:
            self.make_unitary() # shape: (batch_size, tree_size, sample_size+1, basis_size, basis_size)
            
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
        #start = time.time()
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
        #fidelity_max = torch.gather(self.fidelity[:,0,:],1,max_idx) # shape: (batch_size, 1)
        
        
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

        
        
class Universal_2qubit_Gate():
    def __init__(self, **env_params):
        
        # Const @INIT
        ####################################
        self.env_params = env_params
        self.qubit_size = env_params['qubit_size']
        self.basis_size = 2**self.qubit_size
        self.para_size = 4**self.qubit_size -1
        
        self.U2q = None
        self.G = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                               [0.0, -1j, 0.0, 0.0],
                               [0.0, 0.0, -1j, 0.0],
                               [0.0, 0.0, 0.0, 1.0]]).cdouble()
        # G = exp(j*pi/4)*exp(-1j*pi/4 Z*Z)
    
    
    def get_unitary2q(self, theta):
        
        size = theta.size()
        batch_size = size[0]
        tree_size = size[1]
        sample_size = size[2]
        theta_realization = torch.zeros(batch_size, tree_size, sample_size, self.para_size + 9, dtype=torch.complex128)
        ones = torch.ones_like(theta_realization[:,:,:,0])
        # theta's shape: (batch_size, para_size = test_num_paras, sample_size+1, pauli_dim)
        
        
        ###############################################################################################################
        ######### Reference: Realization of a programmable two-qubit quantum processor, Nature Physics (2009).#########
        ###############################################################################################################
        # [theta1, phi1, psi1, theta2, phi2, psi2,
        #   alpha,    0,    0,  3pi/2,    0,    0,
        #    beta, pi/2,    0,  3pi/2,delta,    0,
        #  theta3, phi3, psi3, theta4, phi4, psi4]
        # theta = [theta1, phi1, psi1, theta2, phi2, psi2, alpha, beta, delta, theta3, phi3, psi3, theta4, phi4, psi4]
        # real  = [0,      1,    2,    3,      4,    5,    6,     12,   16,    18,     19,   20,   21,     22,   23  ]
        ##############################################################################################################
        theta_realization[:,:,:,0:6] = theta[:,:,:,0:6].cdouble() # theta1 ~ psi2 >> index 0~5
        theta_realization[:,:,:,6] = theta[:,:,:,6].cdouble() # alpha
        theta_realization[:,:,:,12] = theta[:,:,:,7].cdouble() # beta
        theta_realization[:,:,:,16] = theta[:,:,:,8].cdouble() # delta
        theta_realization[:,:,:,18:] = theta[:,:,:,9:15].cdouble() # theta3 ~ psi4 >> index 9~14
        theta_realization[:,:,:,9] = ones.cdouble()*3*np.pi/2
        theta_realization[:,:,:,13] = ones.cdouble()*np.pi/2
        theta_realization[:,:,:,15] = ones.cdouble()*3*np.pi/2
        
        
        UAB = self.UU(theta_realization[:,:,:,0:6]) # [theta1, phi1, psi1, theta2, phi2, psi2]
        UaR = self.UU(theta_realization[:,:,:,6:12]) # [alpha, 0, 0, 3pi/2, 0, 0]
        UbUd = self.UU(theta_realization[:,:,:,12:18]) # [beta, pi/2, 0, 3pi/2, delta, 0]
        UCD = self.UU(theta_realization[:,:,:,18:24]) # [theta3, phi3, psi3, theta4, phi4, psi4]
        G = self.G[None,None,None,:,:].expand(batch_size,tree_size,sample_size,-1,-1)
        # shape: (batch_size, tree_size, sample_size+1, basis_size, basis_size)
        
        UbUdG = torch.einsum('bgsij,bgsjl -> bgsil',UbUd,G)
        UaRG = torch.einsum('bgsij,bgsjl -> bgsil',UaR,G)
        UaRGUbUdG = torch.einsum('bgsij,bgsjl -> bgsil',UaRG,UbUdG)
        V = torch.einsum('bgsij,bgsjl -> bgsil',G,UaRGUbUdG)
        VCD = torch.einsum('bgsij,bgsjl -> bgsil',V,UCD)
        ABVCD = torch.einsum('bgsij,bgsjl -> bgsil',UAB,VCD) # U2q
        
        return ABVCD
    
    
    ############################################################
    # To calculate tensor product of two 1-qubit universal gates
    ############################################################
    def UU(self,t2q):
        
        size = t2q.size()
        batch_size = size[0]
        tree_size = size[1]
        sample_size = size[2]
        
        U = torch.zeros(batch_size,tree_size,sample_size,self.basis_size,self.basis_size,dtype=torch.complex128)
        t1 = t2q[:,:,:,0:3].cdouble()
        t2 = t2q[:,:,:,3:].cdouble()
        
        u1_00, u1_01, u1_10, u1_11 = self.U1q(t1)
        u2_00, u2_01, u2_10, u2_11 = self.U1q(t2)
        
        U[:,:,:,0,0] = u1_00 * u2_00
        U[:,:,:,0,1] = u1_00 * u2_01
        U[:,:,:,1,0] = u1_00 * u2_10
        U[:,:,:,1,1] = u1_00 * u2_11
        
        U[:,:,:,0,2] = u1_01 * u2_00
        U[:,:,:,0,3] = u1_01 * u2_01
        U[:,:,:,1,2] = u1_01 * u2_10
        U[:,:,:,1,3] = u1_01 * u2_11
        
        U[:,:,:,2,0] = u1_10 * u2_00
        U[:,:,:,2,1] = u1_10 * u2_01
        U[:,:,:,3,0] = u1_10 * u2_10
        U[:,:,:,3,1] = u1_10 * u2_11
        
        U[:,:,:,2,2] = u1_11 * u2_00
        U[:,:,:,2,3] = u1_11 * u2_01
        U[:,:,:,3,2] = u1_11 * u2_10
        U[:,:,:,3,3] = u1_11 * u2_11

        return U

    
    ################################
    # To make 1-qubit universal gate
    ################################
    def U1q(self,t1q):
        
        u00 =  1  * torch.cos(t1q[:,:,:,0]/2) * torch.exp(-1j * t1q[:,:,:,2]/2)
        u01 = -1j * torch.sin(t1q[:,:,:,0]/2) * torch.exp(-1j * t1q[:,:,:,2]/2) * torch.exp(-1j*t1q[:,:,:,1])
        u10 = -1j * torch.sin(t1q[:,:,:,0]/2) * torch.exp( 1j * t1q[:,:,:,2]/2) * torch.exp( 1j*t1q[:,:,:,1])
        u11 =  1  * torch.cos(t1q[:,:,:,0]/2) * torch.exp( 1j * t1q[:,:,:,2]/2)
        
        return u00, u01, u10, u11