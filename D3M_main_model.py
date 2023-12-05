import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI_D3M
from seqnets.SASHIMI import D3M_Sashimi
from D3M import D3M_Linear, D3M_constant, beta_t


class TS_D3M_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.config = config

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.weight1=config['diffusion']['weight1']
        self.weight2=config['diffusion']['weight2']

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        # input_dim = 1 if self.is_unconditional == True else 2
        self.input_dim = config['model']['input_dim']
        if self.is_unconditional:
            assert self.input_dim == 1
        if self.config['diffusion']['type'] == 0:
            self.diffmodel = diff_CSDI_D3M(config, self.input_dim)
        elif self.config['diffusion']['type'] == 1:
            self.diffmodel = D3M_Sashimi(self.config, 
                                         feat_dim=2*self.config['diffusion']['d_feat_dim'],
                                         seq_len=self.config['diffusion']['seq_len'],
                                         cond_in_channels=self.config['diffusion']['side_dim'],
                                         cond_channels=self.config['diffusion']['cond_channels'],
                                         n_layers=2)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]

        func_beta = lambda t: beta_t(t, beta_type=config['diffusion']['beta_type'])
        if config['diffusion']['ddm_type'] == 'constant':
            self.ddm_md = D3M_constant(self.diffmodel, beta=func_beta, 
                                       eps=config['diffusion']['eps'],
                                       weight1=config['diffusion']['weight1'],
                                       weight2=config['diffusion']['weight2'])
        elif config['diffusion']['ddm_type'] == 'linear':
            self.ddm_md = D3M_Linear(self.diffmodel, beta=func_beta, 
                                     eps=config['diffusion']['eps'],
                                     weight1=config['diffusion']['weight1'],
                                     weight2=config['diffusion']['weight2'])

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        l1_sum = 0
        l2_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss, l1, l2 = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t/self.num_steps
            )
            loss_sum += loss.detach()
            l1_sum += l1.detach()
            l2_sum += l2.detach()

        return loss_sum / self.num_steps, l1_sum / self.num_steps, l2_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        
        B, K, L = observed_data.shape
        # import pdb; pdb.set_trace()
        if is_train != 1:
            t = (torch.ones(B) * set_t).to(observed_data.device)
        else:
            t = torch.rand((B,),).to(observed_data.device)

        noise = torch.randn_like(observed_data)
        noisy_data = self.ddm_md.get_noisy_x(observed_data, t, noise)

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask) #
        target_mask = observed_mask - cond_mask

        phi_theta, eps_theta = self.diffmodel(total_input, t, side_info)
        phi = self.ddm_md.get_phi(observed_data)

        if self.config['diffusion']['ddm_type'] == 'constant':
            loss_phi_terms = [((phi_term - hat_phi_term)* target_mask) ** 2 for 
                              phi_term, hat_phi_term in zip(phi, phi_theta)]
        elif self.config['diffusion']['ddm_type'] == 'linear':
            loss_phi_terms = [((phi[0] - phi_theta[0]) * target_mask)**2]
        
        loss = 0.
        l1 = 0.
        for term in loss_phi_terms:
            l1 += term
        loss += self.weight1 * l1
        l2 = ((noise - eps_theta) * target_mask) ** 2
        loss += ( l2  * self.weight2)

        num_eval = target_mask.sum()
        loss = loss.sum() / (num_eval if num_eval > 0 else 1)
        l1 = l1.sum() / (num_eval if num_eval > 0 else 1)
        l2 = l2.sum() / (num_eval if num_eval > 0 else 1)
        # loss = l1 + l2 / l2.detach() * l1.detach()
        return loss, l1, l2

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            
            cond_obs_t = (cond_mask * noisy_data).unsqueeze(1)
            if self.input_dim == 2:
                total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
            elif self.input_dim == 3:
                total_input = torch.cat([cond_obs, noisy_target, cond_obs_t], dim=1)
            elif self.input_dim == 4:
                cond_mask = cond_mask.unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target, cond_obs_t, cond_mask], dim=1)

        return total_input
    
    def impute(self, observed_data, cond_mask, side_info, n_samples, 
               denoise=True, clamp=False, use_pred=True):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        step = 1. / self.num_steps
        time_steps = torch.tensor([step]).repeat(self.num_steps)
        if denoise:
            eps = self.config['diffusion']['eps']
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - eps]), torch.tensor([eps])), dim=0)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    # noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_obs = self.ddm_md.get_noisy_x(noisy_obs, t, noise)
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            cur_time = torch.ones((B, ), device=observed_data.device)

            for t, time_step in enumerate(time_steps):

                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[len(time_steps) -1 - t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    # 这里可以搞花活, 比如融合current_sample, 但需要设计怎么搞，比如开始的时候全用cur_sample，
                    # 随着t->0，使用真实的cond_obs_t
                    cond_obs_t = (cond_mask * current_sample).unsqueeze(1) 
                    
                    # noisy_data = self.ddm_md.get_noisy_x(observed_data, cur_time, torch.randn_like(observed_data))
                    # cond_obs_t = (cond_mask * noisy_data).unsqueeze(1)

                    # cond_obs_t = cond_obs_t_1 * ( 1 - cur_time[0]) + cond_obs_t_2 * cur_time[0]

                    
                    if self.input_dim == 2:
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    elif self.input_dim == 3:
                        diff_input = torch.cat([cond_obs, noisy_target, cond_obs_t], dim=1)  # (B,3,K,L)
                    elif self.input_dim == 4:
                        cond_mask_new = cond_mask.unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target, cond_obs_t, cond_mask_new], dim=1)  # (B,4,K,L)

                s = torch.full((B,), time_step, device=diff_input.device)
                if t == time_steps.shape[0] - 1:
                    s = cur_time
                # denoise_par = self.get_denoise_par(cur_time, *args, **kwargs)
                phi_theta, eps_theta = self.diffmodel(diff_input, cur_time, side_info)
                if use_pred:
                    sample_start = self.ddm_md.pred_x_start(current_sample, eps_theta, phi_theta, cur_time)
                    if clamp:
                        sample_start.clamp_(-1., 1.)
                    if self.config['diffusion']['ddm_type'] == 'constant':
                        phi_new = (-sample_start,)
                    elif self.config['diffusion']['ddm_type'] == 'linear':
                        # phi_new = (phi_theta[0], sample_start / 2) # 注意如果上面phi变了，这里要对应更改
                        phi_new = (phi_theta[0], -sample_start / 2) # 注意如果上面phi变了，这里要对应更改
                    current_sample = self.ddm_md.predict_xtm1_xt(current_sample, phi_new, eps_theta, cur_time, s)
                else:
                    current_sample = self.ddm_md.predict_xtm1_xt(current_sample, phi_theta, eps_theta, cur_time, s)

                cur_time = cur_time - s
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data, # [bs, feat_dim, length]
            observed_mask, # [bs, feat_dim, length]
            observed_tp, # [bs, length]
            gt_mask, # [bs, feat_dim, length]
            for_pattern_mask, # [bs, feat, length]
            _,
        ) = self.process_data(batch)
        
        # import pdb; pdb.set_trace()
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask) # [bs, feat_dim, length]

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples, use_pred=True, denoise=True, clamp=True):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples, 
                                  use_pred=use_pred,
                                  denoise=denoise, 
                                  clamp=clamp)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class TS_D3M_PM25(TS_D3M_base):
    def __init__(self, config, device, target_dim=36):
        super(TS_D3M_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class TS_D3M_Physio(TS_D3M_base):
    def __init__(self, config, device, target_dim=35):
        super(TS_D3M_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
