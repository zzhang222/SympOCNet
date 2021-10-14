#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:18:56 2021

@author: zen
"""
import learner as ln
import numpy as np
import torch
from learner.utils import mse, grad
from time import perf_counter

class SPNN(ln.nn.LossNN):
    '''NN for solving the optimal control of shortest path with obstacles
    '''
    def __init__(self, dim, layers, width, activation, ntype, l, eps, lam, C, add_dim, rho, add_loss, update_lagmul_freq, dtype, device):
        super(SPNN, self).__init__()
        self.dim = dim
        self.ntype = ntype          # (LA/G)SympNet or FNN
        self.dtype = dtype
        self.device = device
        self.l = l                  # hyperparameter controling the soft penalty
        self.eps = eps              # hyperparameter controling the soft penalty
        self.lam = lam              # weight of the BC
        self.C = C                  # speed limit
        self.add_dim = add_dim      # added dimension
        self.latent_dim = add_dim + dim
        # parameters for Lag mul begins
        # Lagrange multiplier for h in opt ctrl prob. Will be a vector in later update
        self.lag_mul_h = torch.zeros(1,dtype=self.dtype, device=self.device)
        # Lagrange multiplier for boundary condition in training process. NOTE: assume two pts bc
        self.lag_mul_bc = torch.zeros(1, 2,self.dim,dtype=self.dtype, device=self.device) 
        self.rho_h = rho	    # parameter for augmented Lagrangian for h
        self.rho_bc = rho	    # parameter for augmented Lagrangian for bc
        self.update_lagmul_freq = update_lagmul_freq
        self.update_lagmul_count = 0
        self.eta0 = 0.1		    # initial tol for aug lag
        self.etak_h = self.eta0	    # k-th tol for aug lag for h
        self.etak_bc = self.eta0    # k-th tol for aug lag for bc
        # parameters for Lag mul ends

        self.__init_param()
        self.__init_net(layers, width, activation, ntype)
        
    # X['interval'] is num * 1
    def criterion(self, X, y):
        Q = self.params['Qslope'] * X['interval'] + self.params['Qincpt']
        P = 0.0 * X['interval'] + self.params['Pincpt']
        QP = torch.cat([Q,P], axis = -1).reshape([-1, self.latent_dim * 2])
        qp = self.net(QP)
        H = self.H(qp)  # (trajs*num) *1
        dH = torch.autograd.grad(H, qp, grad_outputs=torch.ones_like(H), create_graph=True)[0]
        grad_output = self.params['Qslope'].repeat([1,QP.shape[0], 1]).reshape([-1, self.latent_dim])
        grad_output1 = torch.cat([grad_output,torch.zeros_like(grad_output)], dim = -1)
        jacob = torch.autograd.functional.jvp(self.net, QP, grad_output1, create_graph=True)[1]
        loss_1 = mse(jacob[:, :self.latent_dim], dH[...,self.latent_dim:])
        loss_2 = mse(jacob[:, self.latent_dim:], -dH[...,:self.latent_dim])
        loss_sympnet = loss_1 + loss_2
        
        loss = loss_sympnet + self.lam * self.bd_loss(X, y)
        # aug Lag: ||max(0, mul - rho * h(q))||^2/ (2*rho)
        loss = loss + torch.sum(torch.relu(self.lag_mul_h - self.rho_h * self.h(qp[...,:self.dim]))**2)/(2*self.rho_h) # augmented Lagrangian
        # loss for bd
        y_m_bdq = y['bd'] - self.predict_q(X['bd'])
        loss = loss + torch.nn.MSELoss(reduction='sum')(self.lag_mul_bc, self.rho_bc * y_m_bdq)/(2*self.rho_bc)
        return loss
    
    # MSE loss of bdry condition
    def bd_loss(self, X, y):
        bdq = self.predict_q(X['bd'])
        return mse(bdq, y['bd'])
    
    # prediction without added dims
    def predict(self, t, returnnp=False):
        Q = self.params['Qslope'] * t + self.params['Qincpt']
        P = 0.0 * t + self.params['Pincpt']
        QP = torch.cat([Q,P], dim = -1)
        qp = self.net(QP)
        q = qp[...,:self.dim]
        p = qp[...,self.latent_dim:self.latent_dim+self.dim]
        qp = torch.cat([q,p], dim = -1)
        if returnnp:
            qp = qp.detach().cpu().numpy()
        return qp
    
    # prediction q without added dims
    def predict_q(self, t, returnnp=False):
        Q = self.params['Qslope'] * t + self.params['Qincpt']
        P = 0.0 * t + self.params['Pincpt']
        QP = torch.cat([Q,P], dim = -1)
        qp = self.net(QP)
        q = qp[...,:self.dim]
        if returnnp:
            q = q.detach().cpu().numpy()
        return q
        
    # t is num * 1
    def predict_v(self, t, returnnp=False):
        Q = self.params['Qslope'] * t + self.params['Qincpt']
        P = 0.0 * t + self.params['Pincpt']
        QP = torch.cat([Q,P], axis = -1).reshape([-1, self.latent_dim * 2])
        qp = self.net(QP)    
        grad_output = self.params['Qslope'].repeat([1,QP.shape[0], 1]).reshape([-1, self.latent_dim])
        grad_output1 = torch.cat([grad_output,torch.zeros_like(grad_output)], dim = -1)
        v = torch.autograd.functional.jvp(self.net, QP, grad_output1, create_graph=True)[1][:,:self.dim].reshape([1, -1, self.dim])  # 1 * num * dim
        if returnnp:
            v = v.detach().cpu().numpy()
        return v
    
    # penalty function: if x>l, return -log(x); else return -log(l)+1/2*(((x-2l)/l)^2-1)
    def betal(self, x):
        return torch.sum(torch.where(x > self.l, -torch.log(torch.clamp(x, self.l/2)), - np.log(self.l) + 0.5 * (((x - 2*self.l) / self.l) ** 2 - 1)), dim=0)

    # if qp is (trajs*num) * (2latent_dim), then H is (trajs*num) * 1
    def H(self, qp):
        q = qp[...,:self.dim]
        p = qp[...,self.latent_dim:self.latent_dim + self.dim]
        p_dummy = qp[...,self.latent_dim + self.dim:]
        p2 = torch.sum(p.reshape([-1, self.dim // 2, 2]) ** 2, dim = -1)
        # H1 is for real dimensions: sum over all drones, if |p|<C, return |p|^2/2; else return C|p| - C^2/2
        H1 = torch.sum(torch.where(p2 < self.C ** 2, p2/2, self.C*torch.sqrt(p2) - self.C**2/2), dim = -1, keepdims = True)
        # H2 is negative of the added cost (log penalty of h)
        H2 = - self.eps * self.betal(self.h(q))  # eps * beta_l(h(q))
        # H3 is for dummy variables: |p|^2/2
        H3 = torch.sum(p_dummy ** 2, dim = -1, keepdims = True) / 2
        return H1 + H2 + H3

    def update_lag_mul(self, t, bdt, bdy):
        self.update_lagmul_count = self.update_lagmul_count + 1
        # update Lag mul after update_lagmul_freq * print_every steps of training
        if self.update_lagmul_count % self.update_lagmul_freq == 0:
            eta_star = 0.001
            alp, beta = 0.5, 0.5
            tau = 1.1
            # compute constraint h
            q = self.predict_q(t)
            h = self.h(q)
            # compute constraint bc
            bdq = self.predict_q(bdt)
            # update lag_mul for h and bc
            lag_mul_h, lag_mul_bc = self.lag_mul_h, self.lag_mul_bc
            # mul <- max(mul - rho*h, 0)
            new_lag_mul_h = torch.relu(lag_mul_h - self.rho_h * h).detach()
            # mul <- mul + rho*(bdq-y)
            new_lag_mul_bc = (lag_mul_bc + self.rho_bc*(bdq - bdy)).detach()
            # hard constraint: contraint_val == 0
            constraint_h = (new_lag_mul_h - lag_mul_h) / self.rho_h
            constraint_bc = bdq - bdy

            def update_lag_mul_framework(constraint_val, etak, lag_mul, new_lag_mul, rho):
                ret_lag_mul = lag_mul
                ret_etak = etak
                ret_rho = rho
                if torch.max(torch.abs(constraint_val)) < max(eta_star, etak):
                    # update lag mul
                    ret_lag_mul = new_lag_mul
                    ret_etak = etak / (1 + rho ** beta)
                    print('update lag mul step {}, etak {}'.format(torch.max(torch.abs(ret_lag_mul - lag_mul)).item(), ret_etak))
                else:
                    ret_rho = rho * tau
                    ret_etak = self.eta0 / (1+ rho ** alp)
                    print('update rho {}, etak {}'.format(ret_rho, ret_etak))
                return ret_lag_mul, ret_etak, ret_rho

            self.lag_mul_h, self.etak_h, self.rho_h = update_lag_mul_framework(constraint_h, self.etak_h, lag_mul_h, new_lag_mul_h, self.rho_h)
            self.lag_mul_bc, self.etak_bc, self.rho_bc = update_lag_mul_framework(constraint_bc, self.etak_bc, lag_mul_bc, new_lag_mul_bc, self.rho_bc)
    
    # v is ... * dim, L is ... * 1
    def L(self, v): # running cost: sum of |v|^2/2
        return torch.sum(v**2/2, dim=-1, keepdim = True)
    
    def hmin_function(self, t):
        q = self.predict_q(t)
        h = self.h(q)
        hmin = torch.min(h)
        return hmin

    # t is num * 1 and assume t is grid points
    # return size (trajs)
    def value_function(self, t): # compute the value function (ignore constraints)
        dt = (t[-1,0] - t[0,0]) / (list(t.size())[-2] - 1)  # a scalar
        v = self.predict_v(t)   # trajs * num * dim
        L = self.L(v)           # trajs * num * 1
        L[:,0,:] = L[:,0,:]/2
        L[:,-1,:] = L[:,-1,:]/2
        cost = torch.sum(L[...,0], -1) * dt
        return cost
        
    def __init_param(self):
        params = torch.nn.ParameterDict()
        params['Qincpt'] = torch.nn.Parameter(torch.ones((1, 1, self.latent_dim)))
        params['Qslope'] = torch.nn.Parameter(torch.ones((1, 1, self.latent_dim)))
        params['Pincpt'] = torch.nn.Parameter(torch.ones((1, 1, self.latent_dim)))
        self.params = params
        
    def __init_net(self, layers, width, activation, ntype):
        if ntype == 'G':
           self.net = ln.nn.GSympNet(self.latent_dim*2, layers, width, activation)
        elif ntype == 'LA':
           self.net = ln.nn.LASympNet(self.latent_dim*2, layers, width, activation)
        elif ntype == 'FNN':
           self.net = ln.nn.FNN(self.latent_dim*2, self.latent_dim*2, layers, width, activation)
