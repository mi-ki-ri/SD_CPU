import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import time

model_path = "./models/trin"

prompt = """
A portrait of dancing girl, in the rain,
"""

n_prompt = """
man, human, woman, robot,
"""

height = 512
width = 512
num_inference_steps=32
guidance_scale=15
# guidance_scale=25

ITER = 1

pipe = StableDiffusionPipeline.from_pretrained(model_path, local_files_only=True, )

pipe = pipe.to("cpu")
pipe.safety_checker = lambda images, **kwargs: (images, False)

images = pipe(prompt, negative_prompt=n_prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images

with open(f"./dist/{time.time()}.txt", mode="w") as f:
  f.write(f"PROMPT: {prompt}\nN_PROMPT: {n_prompt}\nMODEL:{model_path}\nH:{height}\nW:{width}\nSTEP:{num_inference_steps}\nSCALE:{guidance_scale}")

for image in images:
  image.save(f"./dist/{time.time()}.png")