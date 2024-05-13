import numpy as np
import torch
import time
import argparse
import torchvision.utils as tvu
import matplotlib.pyplot as plt


from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).to(device=DEVICE)

parser = argparse.ArgumentParser("")
parser.add_argument('--dataset', default='probes', choices=['probes', 'sdss'], help='Which dataset?')
parser.add_argument('--milestone', default=750000, dest='milestone', type=int, help='start at this number')
parser.add_argument('--batches', default=105, dest='batches', type=int, help='Number of batches to process.')
parser.add_argument('--batch_size', default=48, dest='batch_size', type=int, help='Batch size.')
parser.add_argument('--timesteps', default=1000, dest='timesteps', type=int, help='Number of timesteps.')
parser.add_argument('--input', '-i', default=None, dest='input', type=str, help='Input Directory')
parser.add_argument('--deg', default='deno', dest='deg', type=str, help='Degradation type')
parser.add_argument('--output', '-o', default='output', dest='output', type=str, help='Output Directory')

parser.add_argument(
        "--sigma_0", type=float, required=True, help="Sigma_0"
    )
parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )
parser.add_argument(
        "--etaB", type=float, default=1, help="Eta_b (before)"
    )
args = parser.parse_args()

diffusion = GaussianDiffusion(
    model,
    timesteps = 1000,
    loss_type = 'l1',
    input_directory='input/'+args.input,
    etaA=args.eta,
    etaB=args.etaB
    
).to(device=DEVICE)


if args.dataset == 'probes':
    trainer = Trainer(
        diffusion,
        './data/probes/',
        logdir = './logs/probes/',
        image_size = 256,
        train_batch_size = 16,
        train_lr = 2e-5,
        train_num_steps = args.milestone,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        num_workers=24,
        rank = [0]
    )

if args.dataset == 'sdss':
    trainer = Trainer(
        diffusion,
        './data/sdss/',
        logdir = './logs/sdss/',
        image_size = 256,
        train_batch_size = 16,
        train_lr = 2e-5,
        train_num_steps = 750001,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        num_workers=32,
        rank = [0]
    )
trainer.load(args.milestone)



def get_H_func():
    ## get degradation matrix ##
        deg = args.deg
        H_funcs = None
        if deg[:2] == 'cs':
            compress_by = int(deg[2:])
            from svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(3, 256, compress_by, torch.randperm(256**2, device=DEVICE), DEVICE)
        elif deg[:3] == 'inp':
            from svd_replacement import Inpainting
            if deg == 'inp_lolcat':
                loaded = np.load("inp_masks/lolcat_extra.npy")
                mask = torch.from_numpy(loaded).to(DEVICE).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_lorem':
                loaded = np.load("inp_masks/lorem3.npy")
                mask = torch.from_numpy(loaded).to(DEVICE).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            else:
                missing_r = torch.randperm(256**2)[:256**2 // 2].to(DEVICE).long() * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            H_funcs = Inpainting(3, 256, missing, DEVICE)
        elif deg == 'deno':
            from svd_replacement import Denoising
            H_funcs = Denoising(3, 256, DEVICE)
        elif deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from svd_replacement import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(DEVICE)
            H_funcs = SRConv(kernel / kernel.sum(), \
                             3, 256, DEVICE, stride = factor)
        elif deg == 'deblur_uni':
            from svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(DEVICE), 3, 256, DEVICE)
        elif deg == 'deblur_gauss':
            from svd_replacement import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(DEVICE)
            H_funcs = Deblurring(kernel / kernel.sum(), 3, 256, DEVICE)
        elif deg == 'deblur_aniso':
            from svd_replacement import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(DEVICE)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(DEVICE)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), 3, 256, DEVICE)
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            from svd_replacement import SuperResolution
            H_funcs = SuperResolution(256)
        elif deg == 'color':
            from svd_replacement import Colorization
            H_funcs = Colorization(256, DEVICE)
        elif deg == 'fft':
            from svd_replacement import Fourier2D
            H_funcs = Fourier2D(256, DEVICE)
        else:
            print("ERROR: degradation type not supported")
            quit()
        args.sigma_0 = 2 * args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = args.sigma_0

        return H_funcs, sigma_0

H_funcs, sigma_0 = get_H_func()

def linear_scale(image):
    min_val_r = np.min(image[0,:,:])
    max_val_r = np.max(image[0,:,:])
    min_val_g = np.min(image[1,:,:])
    max_val_g = np.max(image[1,:,:])
    min_val_z = np.min(image[2,:,:])
    max_val_z = np.max(image[2,:,:])
    
    image[0,:,:] = (image[0,:,:] - min_val_r) / (max_val_r -min_val_r)
    image[1,:,:] = (image[1,:,:] - min_val_g) / (max_val_g -min_val_g)
    image[2,:,:] = (image[2,:,:] - min_val_z) / (max_val_z -min_val_z)

    # set scale from 0 to 1
    image = np.clip(image, 0, 1)
        
    
    return image

def log_scale(image):

    min_val_r = np.min(image[0,:,:])
    min_val_g = np.min(image[1,:,:])
    min_val_z = np.min(image[2,:,:])
    
    image[0,:,:] = np.log(image[0,:,:] - min_val_r + 1)
    image[1,:,:] = np.log(image[1,:,:] - min_val_g + 1)
    image[2,:,:] = np.log(image[2,:,:] - min_val_z + 1)

    return linear_scale(image)

def sigmoid_scale(image):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    min_val_r = np.min(image[0,:,:])
    max_val_r = np.max(image[0,:,:])
    min_val_g = np.min(image[1,:,:])
    max_val_g = np.max(image[1,:,:])
    min_val_z = np.min(image[2,:,:])
    max_val_z = np.max(image[2,:,:])

    #image[0,:,:] = (image[0,:,:] - min_val_r) / (max_val_r -min_val_r)
    #image[1,:,:] = (image[1,:,:] - min_val_g) / (max_val_g -min_val_g)
    #image[2,:,:] = (image[2,:,:] - min_val_z) / (max_val_z -min_val_z)

    image = sigmoid(image)
    



    return image

i = 0
for _ in range(args.batches):
    sampled_batch = diffusion.sample_ddrm(256, timesteps=args.timesteps, batch_size=args.batch_size, H_funcs=H_funcs, sigma_0=sigma_0)
    print(sampled_batch[-1].shape)
    print(type(sampled_batch[-1]))

    for sample in sampled_batch[-1].detach().cpu().numpy():
        np.save(f"inferred/PROBES_2021-10-08/{int(time.time())}_{i:05d}.npy", sample)
        image = linear_scale(sample)
        image_log = log_scale(sample)
        image_sig = sigmoid_scale(sample)
        tvu.save_image(torch.from_numpy(image).float(), f"{args.output}{int(time.time())}_{i:05d}.png")
        tvu.save_image(torch.from_numpy(image_log).float(), f"{args.output}{int(time.time())}_{i:05d}_log.png")
        tvu.save_image(torch.from_numpy(image_sig).float(), f"{args.output}{int(time.time())}_{i:05d}_sig.png")

        print(f"Saved PROBES_2021-10-08/{int(time.time())}_{i:05d}.npy")
        i = i + 1

