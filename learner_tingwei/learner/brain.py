"""
@author: jpzxshi & zen
"""
import os
import time
import numpy as np
import torch

from .nn import LossNN
from .utils import timing, cross_entropy_loss

class Brain:
    '''Runner based on torch.
    '''
    brain = None
    
    @classmethod
    def Init(cls, data, net, criterion, optimizer, lr, iterations, lbfgs_steps, path = None, batch_size=None, 
             print_every=1000, save=False, callback=None, dtype='float', device='cpu', no_test = True):
        cls.brain = cls(data, net, criterion, optimizer, lr, iterations, lbfgs_steps, path, batch_size, 
                         print_every, save, callback, dtype, device, no_test)
    
    @classmethod
    def Run(cls):
        cls.brain.run()
    
    @classmethod
    def Restore(cls):
        cls.brain.restore()
    
    @classmethod
    def Output(cls, data=True, best_model=True, loss_history=True, info=None, **kwargs):
        cls.brain.output(data, best_model, loss_history, info, **kwargs)
    
    @classmethod
    def Loss_history(cls):
        return cls.brain.loss_history
    
    @classmethod
    def Encounter_nan(cls):
        return cls.brain.encounter_nan
    
    @classmethod
    def Best_model(cls):
        return cls.brain.best_model
    
    def __init__(self, data, net, criterion, optimizer, lr, iterations, lbfgs_steps, path, batch_size, 
                 print_every, save, callback, dtype, device, no_test):
        self.data = data
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.iterations = iterations
        self.lbfgs_steps = lbfgs_steps
        self.path = path
        self.batch_size = batch_size
        self.print_every = print_every
        self.save = save
        self.callback = callback
        self.dtype = dtype
        self.device = device
        self.no_test = no_test
        
        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None
        
        self.__optimizer = None
        self.__criterion = None
    
    @timing
    def run(self):
        self.__init_brain()
        print('Training...', flush=True)
        loss_history = []
        for i in range(self.iterations + 1):
            if self.batch_size is not None:
                X_train, y_train = self.data.get_batch(self.batch_size)
                loss = self.__criterion(self.net(X_train), y_train)
            else:
                loss = self.__criterion(self.net(self.data.X_train), self.data.y_train)
            if i % self.print_every == 0 or i == self.iterations:
                if self.no_test:
                    loss_test = loss
                else:
                    loss_test = self.__criterion(self.net(self.data.X_test), self.data.y_test)
                print('{:<9}Train loss: {:<25}Test loss: {:<25}'.format(i, loss.item(), loss_test.item()), flush=True)
                if torch.any(torch.isnan(loss)):
                    try:
                        print('Encountering nan, restore from last saved model.')
                        if self.path is None:
                            self.net = torch.load('model/model{}.pkl'.format(i - self.print_every))
                        else:
                            self.net = torch.load('model/{}/model{}.pkl'.format(self.path, i - self.print_every))
                        self.__init_optimizer()
                        self.__init_criterion()
                        loss = self.__criterion(self.net(self.data.X_train), self.data.y_train)
                    except:
                        self.encounter_nan = True
                        print('Encountering nan, stop training', flush=True)
                        return None
                if self.save:
                    if not os.path.exists('model'): os.mkdir('model')
                    if self.path == None:
                        torch.save(self.net, 'model/model{}.pkl'.format(i))
                    else:
                        if not os.path.isdir('model/'+self.path): os.makedirs('model/'+self.path)
                        torch.save(self.net, 'model/{}/model{}.pkl'.format(self.path, i))
                if self.callback is not None: 
                    output = self.callback(self.data, self.net)
                    loss_history.append([i, loss.item(), loss_test.item(), *output])
                else:
                    loss_history.append([i, loss.item(), loss_test.item()])
            if i < self.iterations:
                self.__optimizer.zero_grad(True)
                loss.backward()
                self.__optimizer.step()
            self.loss_history = np.array(loss_history)
        print('Done!', flush=True)
        return self.loss_history
    
    def restore(self):
        if self.loss_history is not None and self.save == True:
            #best_loss_index = np.argmin(self.loss_history[:, 1])
            # select best model: the h constraint is > min(5* max_h, 0), and cost is min
            constraint_val = self.loss_history[:, 4]
            cost_val = self.loss_history[:, 3]
            max_h = np.amax(constraint_val)  # the max constraint value
            h_tol = min([5.0 * max_h, 0.0])  # if max_h > 0, select h s.t. h>=0; otherwise, select h >= 5 * max_h
            max_cost = np.amax(cost_val)  # the max cost
            max_cost_arr = 0.0 * cost_val + max_cost
            selected_cost = np.where(constraint_val > h_tol, cost_val, max_cost_arr)  # if constraint is ok, select cost; otherwise, put a large number
            min_index = selected_cost.shape[0] // 10 * 7
            best_loss_index = np.argmin(selected_cost[min_index:]) + min_index
            
            iteration = int(self.loss_history[best_loss_index, 0])
            loss_train = self.loss_history[best_loss_index, 1]
            loss_test = self.loss_history[best_loss_index, 2]
            print('Best model at iteration {}:'.format(iteration), flush=True)
            print('Train loss:', loss_train, 'Test loss:', loss_test, flush=True)
            if self.path == None:
            	self.best_model = torch.load('model/model{}.pkl'.format(iteration))
            else:
                self.best_model = torch.load('model/{}/model{}.pkl'.format(self.path,iteration))
        else:
            raise RuntimeError('restore before running or without saved models')
        from torch.optim import LBFGS
        optim = LBFGS(self.best_model.parameters(), history_size=100,
                        max_iter=self.lbfgs_steps,
                        tolerance_grad=1e-08, tolerance_change=1e-09,
                        line_search_fn="strong_wolfe")
        self.it = 0
        def closure():
            if torch.is_grad_enabled():
                optim.zero_grad(True)
            loss = self.best_model.criterion(self.best_model(self.data.X_train), self.data.y_train)
            if self.no_test:
                loss_test = loss
            else:
                loss_test = self.best_model.criterion(self.best_model(self.data.X_test), self.data.y_test)
            # print('Train loss: {:<25}Test loss: {:<25}'.format(loss.item(), loss_test.item()), flush=True)
            it = self.it + 1
            if it % self.print_every == 0 or it == self.lbfgs_steps:
                print('L-BFGS|| It: %05d, Loss: %.4e, Test: %.4e' % 
                          (it, loss.item(), loss_test.item()))
            self.it = it
            if loss.requires_grad:
                loss.backward()
            return loss
        if self.lbfgs_steps > 0:
            optim.step(closure)
        print('Done!', flush=True)
        return self.best_model
    
    def output(self, data, best_model, loss_history, info, **kwargs):
        if self.path is None:
            path = './outputs/' + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        else:
            path = './outputs/' + self.path
        if not os.path.isdir(path): os.makedirs(path)
        if data:
            def save_data(fname, data):
                if isinstance(data, dict):
                    np.savez_compressed(path + '/' + fname, **data)
                else:
                    np.save(path + '/' + fname, data)
            save_data('X_train', self.data.X_train_np)
            save_data('y_train', self.data.y_train_np)
            save_data('X_test', self.data.X_test_np)
            save_data('y_test', self.data.y_test_np)
        if best_model:
            torch.save(self.best_model, path + '/model_best.pkl')
        if loss_history:
            np.savetxt(path + '/loss.txt', self.loss_history)
        if info is not None:
            with open(path + '/info.txt', 'w') as f:
                for key, arg in info.items():
                    f.write('{}: {}\n'.format(key, str(arg)))
        for key, arg in kwargs.items():
            np.savetxt(path + '/' + key + '.txt', arg)
    
    def __init_brain(self):
        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None
        self.data.device = self.device
        self.data.dtype = self.dtype
        self.net.device = self.device
        self.net.dtype = self.dtype
        self.__init_optimizer()
        self.__init_criterion()
    
    def __init_optimizer(self):
        if self.optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
    
    def __init_criterion(self):
        if isinstance(self.net, LossNN):
            self.__criterion = self.net.criterion
            if self.criterion is not None:
                import warnings
                warnings.warn('loss-oriented neural network has already implemented its loss function')
        elif self.criterion == 'MSE':
            self.__criterion = torch.nn.MSELoss()
        elif self.criterion == 'CrossEntropy':
            self.__criterion = cross_entropy_loss
        else:
            raise NotImplementedError
