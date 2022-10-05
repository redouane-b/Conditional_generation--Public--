import torch
import numpy as np
from vae import VAE
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor

ffhq_shift = -112.8666757481
ffhq_scale = 1. / 69.8478027

def normalize_ffhq_input(x):
    return (x + ffhq_shift) * ffhq_scale

    
class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

def filter_state_dict(state_dict, pattern):
    """
    filter_state_dict({'encoder.block_enc1' : ...,'encoder.block_enc2' : ..., 'decoder.block_dec'}, 'encoder') = {'block_enc1' : ..., 'block_enc2' : ..., }
    """
    # remove pattern + the next character '.' to the key
    return {k[len(pattern)+1:] : v for k, v in state_dict.items() if k.startswith(pattern)}


def load_vdvae(conf_path, conf_name, state_dict_path=None, map_location=None):
    H = Hyperparams()
    with open(conf_path) as file:
        H.update(yaml.safe_load(file)[conf_name])
    vae = VAE(H)
    if state_dict_path:
        if map_location:
            state_dict = torch.load(state_dict_path, map_location=map_location)
        else:
            state_dict = torch.load(state_dict_path)
        vae.load_state_dict(state_dict)
    return vae

def save_image_tensor(t, path):
    """
    save a uint8 torch image tensor to path
    """
    im = Image.fromarray(t.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    im.save(path)

def load_image_tensor(path):
    """
    load a image as a torch tensor of type uint8 and shape (1,C,H,W)
    """
    img = Image.open(path)
    t = torch.tensor(np.asarray(img)).permute(2, 0, 1).unsqueeze(0)
    return t

