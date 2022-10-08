from re import I
import torch

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import wandb
import warnings
from tacvis.lightning_modules import ContrastiveModule
import yaml


warnings.filterwarnings("ignore")

project_name = 'GuideN_rgb'

# wandb.init(mode="disabled")
wandb.init(entity='luv',project='tacvis-diffusion',name=project_name) 

# model = Unet(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8)
# ).cuda()
with open('../config/train.yaml', 'r') as stream:
    params = yaml.safe_load(stream)

if params['context']:
    encoder_dir = params['encoder_dir']
    with open(f'{encoder_dir}/params.yaml', 'r') as stream:
        encoder_params = yaml.safe_load(stream)
    both_encoder = ContrastiveModule.load_from_checkpoint(params["encoder_ckpt"],strict=False).eval().cuda()
    if params["rgb"]:
        encoder = both_encoder.rgb_enc
    else:
        encoder = both_encoder.tac_enc
            
    model = Unet(
        dim = 256,
        dim_mults = (1, 2, 4, 8),
        latent_dim = encoder_params["feature_dim"],
    ).cuda()

else:
    encoder =  None
    model = Unet(
        dim = 256,
        dim_mults = (1, 2, 4, 8),
    ).cuda()


diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
).cuda()
trainer = Trainer(
    diffusion,
    '/raid/jkerr/tac_vision/data/ur_data/images_rgb',
    results_folder = f'{project_name}_results',
    train_batch_size=16,
    train_lr=8e-5,
    train_num_steps=1000000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False,  # turn on mixed precision
    save_and_sample_every = 1000,
    context_encoder = encoder
)
# trainer.load(78)


wandb.watch(diffusion, log_freq=100, log="all")
trainer.train()
