import numpy as np
from scipy import special
from scipy import interpolate
from scipy import optimize
import argparse
import torch
import matplotlib.pyplot as plt

class PSmethod :
    def __init__(self, time, num_nodes, dim, net, x_init, x_term, traj_index):
        self.num_nodes = num_nodes # assume num_nodes is the same for all segs
        self.dim = dim
        self.num_segs = len(time) - 1
        self.tau = self._nodes_LGL(num_nodes)
        self.w = self._weight_LGL(num_nodes) # unscaled integration operator
        self.D = self._differentiation_matrix_LGL(num_nodes) # unscaled differential operator
        self.time_ls = []
        for i in range(self.num_segs):
            for j in range(self.num_nodes):
                self.time_ls.append((time[i+1] - time[i]) * (self.tau[j] + 1.0) / 2.0 + time[i])
        self.time_endpts = time
        self.time_all_nodes = np.asarray(self.time_ls)
        self.net = net  # use net to access the h function
        self.x_init = x_init # initial position
        self.x_term = x_term # terminal position

        # initialization
        tensor_t = torch.tensor(self.time_all_nodes[:,None], dtype=self.net.dtype, device=self.net.device)
        x_nn = net.predict_q(tensor_t, True)[traj_index]
        u_nn = net.predict_v(tensor_t, True)[traj_index]
        self.initial_x = x_nn.reshape([-1])
        self.initial_u = u_nn.reshape([-1])
        self.initial_xu = np.concatenate([self.initial_x, self.initial_u])
        
    def solve(self, maxiter):
        # xu is 1D array: first half is x (state), and the second half is u (control)
        # x is flattened nd array, and x[i,j,k] correponds to i-th segment, j-th node, k-th dim
        # u is the same: u[i,j,k] gives control at i-th seg, j-th node, k-th dim
        def equation_constraint(xu, D, num_segs, dim, time_end_pts, x_init, x_term):
            num_nodes = D.shape[0]
            u0_ind = num_segs * num_nodes * dim # u's starting index
            assert len(xu) == 2* u0_ind, "xu length error"
            x = xu[:u0_ind].reshape([num_segs, num_nodes, dim])
            u = xu[u0_ind:].reshape([num_segs, num_nodes, dim])
            # x'=u: (num_segs * num_nodes * dim)
            # err1 <- D @ x - u * (t1-t0)/2 (note the scaling of D)
            err1 = np.matmul(D, x) - np.multiply(u, time_end_pts[1:,None,None] - time_end_pts[:-1,None,None])/2.0
            err1 = err1.reshape([-1])
            # continuity of x: (num_segs - 1) * dim
            # start of next x - end of this x
            err2 = x[1:, 0, :] - x[:-1, -1, :]
            err2 = err2.reshape([-1])
            # x(t0) = x0, x(t1) = x1: 2 * dim
            err3 = (x[0,0,:] - x_init).reshape([-1])  # x(t0)-x0
            err4 = (x[-1,-1,:] - x_term).reshape([-1]) # x(t1)-x1
            return np.concatenate([err1, err2, err3, err4], axis = 0)

        # compute jacobian of equation_constraint
        # return #constraints * #variables
        def equation_constraint_jacob(xu, D, num_segs, dim, time_end_pts, x_init, x_term):
            num_nodes = D.shape[0]
            u0_ind = num_segs * num_nodes * dim # u's starting index, this is also len(x) and len(u)
            assert len(xu) == 2* u0_ind, "xu length error"
            x = xu[:u0_ind].reshape([num_segs, num_nodes, dim])
            u = xu[u0_ind:].reshape([num_segs, num_nodes, dim])
            # x'=u: (num_segs * num_nodes * dim)
            # err1 <- D @ x - u * (t1-t0)/2 (note the scaling of D)
            jacob1_x = np.zeros((num_segs, num_nodes, dim, num_segs, num_nodes, dim))
            jacob1_u = np.zeros((num_segs, num_nodes, dim, num_segs, num_nodes, dim))
            for i in range(num_segs):
                for k in range(dim):
                    jacob1_x[i,:,k,i,:,k] = D
            for i in range(num_segs):
                for j in range(num_nodes):
                    for k in range(dim):
                        jacob1_u[i,j,k,i,j,k] = -(time_end_pts[i+1] - time_end_pts[i])/2.0
            jacob1 = np.concatenate([jacob1_x.reshpe([u0_ind, u0_ind]), jacob1_u.reshape([u0_ind, u0_ind])], axis = -1)
            # continuity of x: (num_segs - 1) * dim
            # start of next x - end of this x
            jacob2 = np.zeros((num_segs-1, dim, num_segs*2, num_nodes, dim))
            for i in range(num_segs-1):
                for k in range(dim):
                    jacob2[i,k,i+1,0,k] += 1.0
                    jacob2[i,k,i,-1,k] -= 1.0
            jacob2 = jacob2.reshape([(num_segs-1)*dim, u0_ind * 2])
            # x(t0) = x0, x(t1) = x1: 2 * dim
            jacob3 = np.zeros((2*dim, 2* u0_ind))
            jacob3[:dim, :dim] = np.eye(dim)
            jacob3[dim:, -dim:] = np.eye(dim)
            return np.concatenate([jacob1, jacob2, jacob3], axis = 0)

        # xu is 1D array: first half is x (state), and the second half is u (control)
        # x is flattened nd array, and x[i,j,k] correponds to i-th segment, j-th node, k-th dim
        # u is the same: u[i,j,k] gives control at i-th seg, j-th node, k-th dim
        def nonneg_constraint(xu, net, num_segs, num_nodes) :
            dim = net.dim
            u0_ind = num_segs * num_nodes * dim # u's starting index
            assert len(xu) == 2* u0_ind, "xu length error"
            x = xu[:u0_ind].reshape([num_segs, num_nodes, dim])
            u = xu[u0_ind:].reshape([num_segs, num_nodes, dim])
            # h(x) : (num_segs * num_nodes)
            err = net.h_np(x).reshape([-1]) # note: x is num_segs * num_nodes * dim
            # bound for u
            err1 = (u + net.C).reshape([-1])
            err2 = (net.C - u).reshape([-1])
            '''
            # bound for x: not used for now
            bd_x = 5.0
            err3 = (x + bd_x).reshape([-1])
            err4 = (bd_x - x).reshape([-1])
            #ret = np.concatenate([err, err1, err2, err3, err4], axis = 0)
            '''
            ret = np.concatenate([err, err1, err2], axis = 0)
            return ret

        '''
        # TODO: is h differentiable?
        def nonneg_constraint_jacob(xu, net, num_segs, num_nodes) :
            pass
        '''

        # cost = int_{t0}^{t1} |u|^2/2 dt up to scaling
        def cost(xu, weight, num_segs, dim):
            num_nodes = weight.shape[0]
            u0_ind = num_segs * num_nodes * dim # u's starting index
            assert len(xu) == 2* u0_ind, "xu length error"
            u = xu[u0_ind:].reshape([num_segs, num_nodes, dim])
            cost = np.sum(u**2/2, axis = -1)
            cost = np.multiply(cost, weight)
            return np.sum(cost) / (2*num_segs)

        # return gradient of cost wrt (x,u). Size: (#varibles)
        def cost_grad(xu, weight, num_segs, dim):
            num_nodes = weight.shape[0]
            u0_ind = num_segs * num_nodes * dim # u's starting index
            assert len(xu) == 2* u0_ind, "xu length error"
            u = xu[u0_ind:].reshape([num_segs, num_nodes, dim])
            grad = np.multiply(u, weight[None,:,None])
            return np.concatenate([np.zeros(u0_ind), grad.reshape([-1])])

        cons = ({'type': 'eq',
                 'fun': equation_constraint,
                 'args': (self.D, self.num_segs, self.dim, self.time_endpts, self.x_init, self.x_term)},
                {'type': 'ineq',
                 'fun': nonneg_constraint,
                 'args': (self.net, self.num_segs, self.num_nodes)}
                )

        ftol = 1e-6

        print('max initial eq constraint val:')
        print(np.amax(np.abs(equation_constraint(self.initial_xu, self.D, self.num_segs, self.dim, self.time_endpts, self.x_init, self.x_term))))
        print('min initial ineq constraint val:')
        print(np.amin(nonneg_constraint(self.initial_xu, self.net, self.num_segs, self.num_nodes)))

        res = optimize.minimize(cost, self.initial_xu, args=(self.w, self.num_segs, self.dim),
                                constraints=cons, jac=cost_grad, method='SLSQP',
                                options={"disp": True, "maxiter": maxiter, "ftol": ftol})
        self.opt_xu = res.x
        print('max PS eq constraint val:')
        print(np.amax(np.abs(equation_constraint(res.x, self.D, self.num_segs, self.dim, self.time_endpts, self.x_init, self.x_term))))
        print('min PS ineq constraint val:')
        print(np.amin(nonneg_constraint(res.x, self.net, self.num_segs, self.num_nodes)))
        
    def get_x(self):
        x_len = self.num_segs * self.num_nodes * self.dim # length of x_arr
        assert len(self.opt_xu) == 2* x_len, "xu length error"
        x = self.opt_xu[:x_len].reshape([-1,self.dim])
        return x
        
    def get_initial_x(self):
        x_len = self.num_segs * self.num_nodes * self.dim # length of x_arr
        assert len(self.initial_x) == x_len, "initial_x length error"
        x = self.initial_x.reshape([-1,self.dim])
        return x

    def _nodes_LGL(self, n):
        """ Legendre-Gauss-Lobatto(LGL) points"""
        roots, weight = special.j_roots(n-2, 1, 1)
        nodes = np.hstack((-1, roots, 1))
        return nodes

    def _weight_LGL(self, n):
        """ Legendre-Gauss-Lobatto(LGL) weights."""
        nodes = self._nodes_LGL(n)
        w = np.zeros(0)
        for i in range(n):
            w = np.append(w, 2/(n*(n-1)*self._LegendreFunction(nodes[i], n-1)**2))
        return w

    def _differentiation_matrix_LGL(self, n):
        """ Legendre-Gauss-Lobatto(LGL) differentiation matrix."""
        tau = self._nodes_LGL(n)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = self._LegendreFunction(tau[i], n-1) \
                              / self._LegendreFunction(tau[j], n-1) \
                              / (tau[i] - tau[j])
                elif i == j and i == 0:
                    D[i, j] = -n*(n-1)*0.25
                elif i == j and i == n-1:
                    D[i, j] = n*(n-1)*0.25
                else:
                    D[i, j] = 0.0
        return D

    def _LegendreFunction(self, x, n):
        Legendre, Derivative = special.lpn(n, x)
        return Legendre[-1]

    def _LegendreDerivative(self, x, n):
        Legendre, Derivative = special.lpn(n, x)
        return Derivative[-1]
