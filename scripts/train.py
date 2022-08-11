import torch

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import wandb
import warnings
warnings.filterwarnings("ignore")

project_name = 't10_noaug'

# wandb.init(mode="disabled")
wandb.init(entity='luv',project='tacvis-diffusion',name=project_name) 

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    '/home/jkerr/tac_vision/data/small_diffusion_training_data',
    results_folder = f'{project_name}_results',
    train_batch_size=9,
    train_lr=3e-5,
    train_num_steps=100000,  # total training steps
    gradient_accumulate_every=5,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
    save_and_sample_every = 1000
)


trainer.train()
