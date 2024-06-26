import math
import copy
import os
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from astropy.io import fits
from PIL import Image

import numpy as np
from tqdm import tqdm
from einops import rearrange

from time import time
import imageio.v3 as iio
import torchvision.utils as tvu
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#   _   _      _
#  | | | | ___| |_ __   ___ _ __ ___
#  | |_| |/ _ \ | '_ \ / _ \ '__/ __|
#  |  _  |  __/ | |_) |  __/ |  \__ \
#  |_| |_|\___|_| .__/ \___|_|  |___/
#               |_|

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g

#   ____        _ _     _ _               _     _            _
#  | __ ) _   _(_) | __| (_)_ __   __ _  | |__ | | ___   ___| | _____
#  |  _ \| | | | | |/ _` | | '_ \ / _` | | '_ \| |/ _ \ / __| |/ / __|
#  | |_) | |_| | | | (_| | | | | | (_| | | |_) | | (_) | (__|   <\__ \
#  |____/ \__,_|_|_|\__,_|_|_| |_|\__, | |_.__/|_|\___/ \___|_|\_\___/
#                                 |___/

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

#   _   _            _                         _      _
#  | | | |_ __   ___| |_   _ __ ___   ___   __| | ___| |
#  | | | | '_ \ / _ \ __| | '_ ` _ \ / _ \ / _` |/ _ \ |
#  | |_| | | | |  __/ |_  | | | | | | (_) | (_| |  __/ |
#   \___/|_| |_|\___|\__| |_| |_| |_|\___/ \__,_|\___|_|
# 

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = dim),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = dim),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

#    ____                     _                   _ _  __  __           _
#   / ___| __ _ _   _ ___ ___(_) __ _ _ __     __| (_)/ _|/ _|_   _ ___(_) ___  _ __
#  | |  _ / _` | | | / __/ __| |/ _` | '_ \   / _` | | |_| |_| | | / __| |/ _ \| '_ \
#  | |_| | (_| | |_| \__ \__ \ | (_| | | | | | (_| | |  _|  _| |_| \__ \ | (_) | | | |
#   \____|\__,_|\__,_|___/___/_|\__,_|_| |_|  \__,_|_|_| |_|  \__,_|___/_|\___/|_| |_|
#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

class GaussianDiffusion(nn.Module):
    def __init__(
        self, 
        denoise_fn, 
        *,
        image_size = 256,
        channels = 3,
        timesteps = 1000, 
        loss_type = 'l1', 
        betas = None,
        input_directory,
        etaA,
        etaB,
        etaC
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        # def sigmoid(x):
        #     return 1 / (np.exp(-x) + 1)

        # betas = np.linspace(-6, 6, 1000)
        # betas = sigmoid(betas) * (0.0001 - 1) + 1

        # betas = np.linspace(
        #     0.0001, 0.02, 1000, dtype=np.float64
        # )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
    #    self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.input_directory = input_directory
        self.input_files = os.listdir(self.input_directory)
        print(self.input_files)
        self.etaA = etaA
        self.etaB = etaB
        self.etaC = etaC




    
    def load_n_images(self, n):
        imgs = np.empty((n, 3 *256 * 256))
        
        for i in range(n):
            if len(self.input_files) == 0:
                print("No more files to input")
            path = self.input_directory + '/' + self.input_files[0]
            print("Loading image: ", path)
            self.input_files = self.input_files[1:]
            img = iio.imread(path)
            

            if self.input_args.duplicate:
                image0 = Image.fromarray(img)
                downsampled_image = image0.resize((128, 128), Image.LANCZOS)
                # get the image as numpy array
                image_temp = np.array(downsampled_image)
                img = np.zeros((256, 256, 3))
                img[:128, :128] = image_temp
                img[:128, 128:] = np.fliplr(image_temp)
                img[128:, :128] = np.flipud(image_temp)
                img[128:, 128:] = np.flipud(np.fliplr(image_temp))


            img = img.transpose((2, 0, 1))
            img = (img / 255.0) * 2. - 1.
            imgs[i] = img.flatten()
            

        return imgs
    

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance





    @torch.no_grad()
    def p_sample_loop_ddrm(self, x, y_0, seq, H_funcs, etaA, etaC, etaB, sigma_0):
        device = self.betas.device

        # facilité
        b = self.betas
        sigma_0 = float(sigma_0)


        singulars = H_funcs.singulars()
        Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
        Sigma[:singulars.shape[0]] = singulars

        U_t_y = H_funcs.Ut(y_0)



        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]
        Sig_inv_U_t_y = U_t_y.clone()
  
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
        inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device).to(singulars.dtype)

        inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init) 
        init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
        init_y[:, large_singulars_index[0]] = (U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)).to(init_y.dtype)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
        remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas



        #setup iteration variables
        x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        print('init_y nan check : ' + str(torch.isnan(init_y).any()))

        print('x nan check : ' + str(torch.isnan(x).any()))
        
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), total=len(seq), desc='sampling loop time step'):
            t = (torch.ones(n, dtype=torch.long) * i).to(device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            # consult the model
            et = self.denoise_fn(xt, t)

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            #variational inference conditioned on y
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            #print('cond_after_lite shape : ',cond_after_lite.shape)
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))
            #print('cond_after shape : ',cond_after.shape)

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
            
            diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

            #missing pixels
            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

            #print('U_t_y shape : ', U_t_y.shape)
            #print('SVt_x0 shape : ', SVt_x0.shape)
            #less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = \
                V_t_x0[:, cond_after] + \
                sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + \
                std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            
            
            #noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = \
                (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

            #aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            #print(f' xt_next nan check: at step {i} ', torch.isnan(xt_next).any())
            #print(f' xt_next inf check: at step {i} ', torch.isinf(xt_next).any())

            #print(f' xt_mod_next nan check: at step {i} ', torch.isnan(xt_mod_next).any())
            #print(f' xt_mod_next inf check: at step {i} ', torch.isinf(xt_mod_next).any())

            #print(f' Vt_xt_mod_next nan check: at step {i} ', torch.isnan(Vt_xt_mod_next).any())
            #print(f' Vt_xt_mod_next inf check: at step {i} ', torch.isinf(Vt_xt_mod_next).any())

            #print(f' x0_preds nan check: at step {i} ', torch.isnan(x0_t).any())
            #print(f' x0_preds inf check: at step {i} ', torch.isinf(x0_t).any())

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            '''   
            x_recon = extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * self.denoise_fn(x, t)

            # previously conditioned on clip_denoised boolean variable
            x_recon.clamp_(-1., 1.)
            model_mean, _, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
            # previously conditioned on repeat_noise boolean variable
            noise = noise_like(x.shape, device, False)
            # mask for t==0
            nonzero_mask = (1 - (t == 0).float()).reshape(n, *((1,) * (len(x.shape) - 1)))
            x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            xs.append(x)
            '''
            


        return xs  
        #return xs, x0_preds
    
    @torch.no_grad()
    def sample_ddrm(self, image_size, timesteps, H_funcs, args, batch_size = 16, sigma_0=0):
        self.device = self.betas.device
        self.input_args = args
        deg = args.deg
        image_size = self.image_size
        channels = self.channels
        
        if args.duplicate:
            noise = torch.randn(
                batch_size,
                channels, 
                128,
                128,
                device=self.device,
            )
            # define empty tensor size 256 by 256
            x = torch.zeros(
                batch_size,
                channels, 
                image_size,
                image_size,
                device=self.device,
            )
            x[:, :, :128, :128] = noise
            x[:, :, :128, 128:] = torch.fliplr(noise)
            x[:, :, 128:, :128] = torch.flipud(noise)
            x[:, :, 128:, 128:] = torch.flipud(torch.fliplr(noise))
        else : 
            x = torch.randn(
                batch_size,
                channels, 
                image_size,
                image_size,
                device=self.device,
            )
        


        x_orig = torch.from_numpy(self.load_n_images(batch_size)).to(self.device).float()

        # y = H(x) + epsilon
        y_0 = H_funcs.H(x_orig)
        y_0 = y_0 +  sigma_0 * ( torch.randn_like(y_0) )

        print('y_0 shape:', y_0.shape)
        pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], channels, image_size, image_size)

        if deg[:6] == 'deblur': pinv_y_0 = y_0.view(y_0.shape[0], channels, image_size, image_size)
        elif deg == 'color': pinv_y_0 = y_0.view(y_0.shape[0], 1, image_size, image_size).repeat(1, 3, 1, 1)
        elif deg[:3] == 'inp': pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1
        elif deg == 'fft_inp': pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1



        print('Saving images')
        for i in range (len(pinv_y_0)):
                # check if pinv_y_0 is complex
                image_to_save = (pinv_y_0[i]+1)*2
                if torch.is_complex(pinv_y_0[i]):
                    image_to_save = (pinv_y_0[i]).real
                tvu.save_image(image_to_save, f"{args.output}defaced_{i}.png")
        print('Saved Images')


        skip = 1000//timesteps
        seq = range(0, 1000, skip)

        retval = self.p_sample_loop_ddrm(x=x, y_0=y_0, seq=seq, H_funcs=H_funcs, etaA=self.etaA,
                                       etaC=self.etaC, etaB=self.etaB, sigma_0=2*sigma_0)
        
        for i in range(len(x_orig)):
            orig =  x_orig[-1][i]
            mse = torch.mean((x[-1][i].to(self.device) - orig) ** 2)
            #psnr = 10 * torch.log10(1 / mse)

            # cosine similarity
            print(f"MSE: {mse}")

        
        return retval

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device

        #1 model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)

        #2 x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))
        
        
        x_recon = extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * self.denoise_fn(x, t)
        #2

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        #1
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        #for i in tqdm(reversed(range(0, 100)), desc='sampling loop time step', total=100):
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):

            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, image_size, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    @torch.no_grad()
    def q_then_p(self, x_start, t, batch_size=16, mask=None):
        device = self.betas.device
        if mask is not None:
            mask = torch.stack([mask] * batch_size)
            x_start = torch.where(mask == True, torch.tensor(-1.0).to("cuda"), x_start)

        if t == 1000:
            zs = torch.randn(x_start.shape, device=device)
        else:
            zs = self.q_sample(x_start, torch.tensor([t] * batch_size).to(device))
        ps = zs
        for i in tqdm(reversed(range(0, t)), desc='domain transfer time step', total=t):
            if mask is not None:
                zs = self.q_sample(x_start, torch.tensor([i] * batch_size).to(device))
                ps = torch.where(mask == False, zs, ps)

            ps = self.p_sample(ps, torch.full((batch_size,), i, device=device, dtype=torch.long))

        #if mask is not None:
        #    ps = self.p_sample(ps, torch.full((batch_size,), i, device=device, dtype=torch.long))
        #    ps = torch.where(mask == False, x_start, ps)

        return ps

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)

#   ____        _                 _          _
#  |  _ \  __ _| |_ __ _ ___  ___| |_    ___| | __ _ ___ ___
#  | | | |/ _` | __/ _` / __|/ _ \ __|  / __| |/ _` / __/ __|
#  | |_| | (_| | || (_| \__ \  __/ |_  | (__| | (_| \__ \__ \
#  |____/ \__,_|\__\__,_|___/\___|\__|  \___|_|\__,_|___/___/
# 

class Galaxies(data.Dataset):
    def __init__(self, folder, image_size, minmaxnorms=(0, 5.5)):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = list(Path(f'{folder}').glob(f'**/*.npy'))
        f = open('paths.txt', 'w')
        for path in self.paths[:200]:
            f.write(f'{path}\n')
        self.min_ = minmaxnorms[0]
        self.max_ = minmaxnorms[1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = np.load(path)

        img = np.clip(img, self.min_, self.max_)
        img = 2*(img - self.min_)/(self.max_ - self.min_) - 1 # A min max norm for all maxima == 5 and minima == 0.0 gals

        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1)
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2)

        img = img.copy()
        return torch.tensor(img)

#   _____          _                        _
#  |_   _| __ __ _(_)_ __   ___ _ __    ___| | __ _ ___ ___
#    | || '__/ _` | | '_ \ / _ \ '__|  / __| |/ _` / __/ __|
#    | || | | (_| | | | | |  __/ |    | (__| | (_| \__ \__ \
#    |_||_|  \__,_|_|_| |_|\___|_|     \___|_|\__,_|___/___/
# 

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 256,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        rank = [0, 1, 2],
        num_workers = 128,
        save_every = 5000,
        sample_every = 5000,
        logdir = './logs',
    ):
        super().__init__()
        self.model = torch.nn.DataParallel(diffusion_model, device_ids=rank)
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_every = save_every
        self.sample_every = sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.logdir = Path(logdir)
        self.logdir.mkdir(exist_ok = True)

        #self.ds = Galaxies(folder, image_size, minmaxnorms=(0, 5.5))
        #self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, num_workers=num_workers))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.logdir / f'{milestone:08d}-model.pt'))

    def load(self, milestone):
        data = torch.load(str(self.logdir / f'{milestone:08d}-model.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):

        t1 = time()
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).to(device=DEVICE)
                while torch.any(~torch.isfinite(data)):
                    print("NAN DETECTED!!")
                    data = next(self.dl).to(device=DEVICE)
                loss = self.model(data).sum()
                t0 = time()
                print(f'{self.step}: {loss.item()}, delta_t: {t0 - t1:.03f}')
                t1 = time()
                with open(str(self.logdir / 'loss.txt'), 'a') as df:
                    df.write(f'{self.step},{loss.item()}\n')
                (loss / self.gradient_accumulate_every).backward()

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.sample_every == 0:
                batches = num_to_groups(18, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.module.sample(self.image_size, batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = torch.flip(all_images, dims=[1]) # map channels correctly for imout
                all_images = all_images + 1
                all_images = list(map(lambda x: (x - x.min())/(x.max() - x.min()), all_images))
                utils.save_image(all_images, str(self.logdir / f'{self.step:08d}-sample.jpg'), nrow=6)

            if self.step != 0 and self.step % self.save_every == 0:
                self.save(self.step)

            self.step += 1

        print('training completed')
