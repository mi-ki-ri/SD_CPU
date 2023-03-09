import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

access_tokens = "" # TOKEN HERE
modeler = "stabilityai"
model_name = "stable-diffusion-2"

pipe = StableDiffusionPipeline.from_pretrained(f"{modeler}/{model_name}", use_auth_token=access_tokens)
pipe.save_pretrained(f"./models/{modeler}_{model_name}")