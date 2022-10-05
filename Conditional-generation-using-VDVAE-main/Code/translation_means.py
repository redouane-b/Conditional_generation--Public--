from ast import Break
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import load_vdvae,load_image_tensor, normalize_ffhq_input
from pathlib import Path
from torchvision.utils import make_grid
from PIL import Image

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning or DeprecationWarning)

#level of latent variables
k=5
src=2
step=2
lim=30

def translation(point,vector,constant):
  point = [p+constant*v for p,v in zip(point,vector)]
  return point

# load vae
vae = load_vdvae(
        conf_path="Code_python/saved_models/confs.yaml",
        conf_name="base",
        state_dict_path="Code_python/saved_models/ffhq256-iter-1700000-model-ema.th",
        map_location='cpu')
vae.eval()

def get_latents(vae, x):
    with torch.no_grad():
        stats = vae.forward_get_latents(x) # [{'z' : ..., 'kl' : ...}]
    latents = [d['z'] for d in stats] # [z1, ..., zL] z1.shape = [1,C,H,W]
    return latents[0:lim]

with open('Code_python/vectors/gender_vector.npy',"rb") as f:
  vector=pickle.load(f)




x = load_image_tensor("Code_python/images/{}.png".format(src))
x_f = normalize_ffhq_input(x)
ZS = get_latents(vae, x_f)
p=ZS






vector.append(0)
vector[0]=0
vector[1]=0



l=[]

with torch.no_grad():
  for i in range(6):
    elem=vae.forward_samples_set_latents(4, p)
    l.append(elem)
    p=translation(p,vector,step)

l = torch.cat(l)
grid = make_grid(l, nrow=3)
grid=grid.permute(1, 2, 0)
grid=grid.detach().cpu().numpy()


outpath = Path("res/").joinpath('image_using_{}_levels_from_im{}_with_step{}.png'.format(k,src,step))
im = Image.fromarray(grid)
im.save(outpath)
 
