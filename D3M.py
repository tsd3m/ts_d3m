import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def beta_t(t, beta_type='sqrt'):
    if beta_type == 'sqrt':
        return t.sqrt()
    elif beta_type == 'linear':
        return t
    elif beta_type == 'square':
        return t**2
    elif beta_type == 'exp':
        return (torch.exp(t) - 1) / (np.exp(1) - 1)
    elif beta_type == 'sin':
        return torch.sin(np.pi / 2 * t)
    else:
        raise NotImplementedError



class D3M_base(nn.Module):
    '''
    D3M: Decomposable Denoise Diffusion Models
    '''
    def __init__(self, net, beta, eps=1e-4, weight1=1, weight2=1):
        super().__init__()
        '''
        net: eps_theta
        beta: function\beta_t, used for N(\alpha_t x_0, \beta_t^2 I)
        eps: used for denoise sampling
        '''
        self.net = net
        self.eps = eps
        self.beta = beta
        self.weight1 = weight1
        self.weight2 = weight2

    def get_denoise_par(self, t, *args, **kwargs):
        res = (t,)
        if args:
            res += args
        if kwargs:
            res += tuple(kwargs.values())
        return res

    def get_H_t(self, t, *phi):
        pass

    def get_phi(self, x_0):
        pass

    def get_beta_t(self, t):
        return self.beta(t)

    def get_noisy_x(self, x_0, t, noise):
        t = t.reshape(t.shape[0], *((1,)*(len(x_0.shape)-1)))
        phi = self.get_phi(x_0=x_0)

        x_t = x_0 + self.get_H_t(t, *phi) + self.get_beta_t(t) * noise
        return x_t

    def forward(self, x_0, *args, **kwargs):
        noise = torch.randn_like(x_0)
        t = torch.rand((x_0.shape[0],),).to(x_0.device)
        x_t = self.get_noisy_x(x_0, t, noise)
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        phi = self.get_phi(x_0)
        phi_theta, eps_theta = self.net(x_t, *denoise_par)
        # loss_phi_terms = [F.mse_loss(phi_term, hat_phi_term) for phi_term, hat_phi_term in zip(phi, phi_theta)]
        # loss = sum(loss_phi_terms) + F.mse_loss(noise, eps_theta)
        loss_phi_terms = [(phi_term - hat_phi_term) ** 2 for
                          phi_term, hat_phi_term in zip(phi, phi_theta)]
        l1 = sum(loss_phi_terms)
        l2 = (noise - eps_theta) ** 2
        loss = self.weight1 * l1 + self.weight2 * l2
        return loss
    
    @torch.no_grad()
    def sample(self, shape, num_steps, device, denoise=True, clamp=True, *args, **kwargs):
        x = torch.randn(shape).to(device)
        x = self.sample_loop(x, num_steps, denoise=denoise,  clamp=clamp, *args, **kwargs)
        return x

    
    def predict_xtm1_xt(self, xt, phi, noise, t, s):
        '''
        Input:
            xt: 
            phi: 
            noise: 
            t: 
            s: 
        Return:
            x_{t-1}
        '''
        t = t.reshape(xt.shape[0], *((1,) * (len(xt.shape) - 1) ))
        s = s.reshape(xt.shape[0], *((1,) * (len(xt.shape) - 1) ))

        beta_t = self.get_beta_t(t)
        beta_t_s = self.get_beta_t(t-s)

        mean = xt + self.get_H_t(t-s, *phi) - self.get_H_t(t, *phi) - (beta_t**2 - beta_t_s**2) / beta_t * noise
        sigma = (beta_t_s / beta_t) * ((beta_t**2 - beta_t_s**2)).sqrt()
        eps = torch.randn_like(mean, device=xt.device)
        return mean + sigma * eps
    
    def pred_x_start(self, x_t, noise, phi, t):
        t = t.reshape(t.shape[0], *((1,)*(len(x_t.shape)-1)))
        x_0 = x_t - self.get_H_t(t, *phi) - self.get_beta_t(t) * noise
        return x_0

    
    def sample_loop(self, x, num_steps, denoise=True,  clamp=True, *args, **kwargs):
        bs = x.shape[0]

        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs, ), device=x.device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                s = cur_time

            denoise_par = self.get_denoise_par(cur_time, *args, **kwargs)
            phi_theta, eps_theta = self.net(x, *denoise_par)

            x = self.predict_xtm1_xt(x, phi_theta, eps_theta, cur_time, s)
            if clamp:
                x.clamp_(-1., 1.)
            cur_time = cur_time - s
       
        return x


class D3M_constant(D3M_base):
    def get_H_t(self, t, *phi):
        C = phi[0]
        t = t.reshape(t.shape[0], *((1,)*(len(C.shape)-1)))
        return t * C

    def get_phi(self, x_0):
        return (- x_0,)    
 
    def sample_loop(self, x, num_steps, denoise=True,  clamp=True, *args, **kwargs):
        bs = x.shape[0]

        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs, ), device=x.device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                s = cur_time

            denoise_par = self.get_denoise_par(cur_time, *args, **kwargs)
            phi_theta, eps_theta = self.net(x, *denoise_par)

            x_0 = self.pred_x_start(x, eps_theta, phi_theta, cur_time)
            if clamp:
                x_0.clamp_(-1., 1.)
            phi_new = (-x_0, )

            x = self.predict_xtm1_xt(x, phi_new, eps_theta, cur_time, s)
            cur_time = cur_time - s
        
        return x
    

class D3M_Linear(D3M_base):

    def get_H_t(self, t, *phi):
        a, b = phi[0], phi[1]
        t = t.reshape(t.shape[0], *((1,)*(len(a.shape)-1)))
        return a / 2 * t ** 2 + b * t

    def get_phi(self, x_0):
        '''
        you can set these two parameters to determine the forward process.
        '''
        # a = -3 * x_0
        # b = x_0 / 2
        a = - x_0
        b = - x_0 / 2
        return (a, b)
    
    def forward(self, x_0, *args, **kwargs):
        noise = torch.randn_like(x_0)
        t = torch.rand((x_0.shape[0],),).to(x_0.device)
        x_t = self.get_noisy_x(x_0, t, noise)

        denoise_par = self.get_denoise_par(t, *args, **kwargs)     
        phi_theta, eps_theta = self.net(x_t, *denoise_par)

        phi = self.get_phi(x_0)
        loss_phi_term = F.mse_loss(phi[0], phi_theta[0])
        loss = self.weight1 * loss_phi_term + self.weight2 * F.mse_loss(noise, eps_theta)
        return loss
    
    def pred_x_start(self, x_t, noise, phi, t):
        t = t.reshape(t.shape[0], *((1,)*(len(x_t.shape)-1)))
        if t[0] == 1.:
            # print('aaa')
            den = 1 - t + self.eps
        else:
            den = 1 - t
        x_0 = ( x_t - self.get_beta_t(t) * noise + phi[0] * t * (1 - t) / 2 ) / den
        return x_0
 
    def sample_loop(self, x, num_steps, denoise=True,  clamp=True, *args, **kwargs):
        bs = x.shape[0]

        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs, ), device=x.device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                s = cur_time

            denoise_par = self.get_denoise_par(cur_time, *args, **kwargs)
            phi_theta, eps_theta = self.net(x, *denoise_par)

            x_0 = self.pred_x_start(x, eps_theta, phi_theta, cur_time)
            if clamp:
                x_0.clamp_(-1., 1.)
            phi_new = (phi_theta[0], -x_0 / 2)

            x = self.predict_xtm1_xt(x, phi_new, eps_theta, cur_time, s)
            cur_time = cur_time - s
        
        return x