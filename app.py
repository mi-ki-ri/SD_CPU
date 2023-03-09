import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import time

model_path = "./models/trin"

prompt = """
((contraposto)), color pencil sketch of a ((walking)) Fashon model girl, ((landscape)) shot, beautiful face, detailed face, art by akihiko yoshida, solid background,
"""

n_prompt = """
color pencil, sketchbook,
2girls, ugly, poorly drawn hands, poorly drawn feet, poorly drawn face,
out of frame, body out of frame, bad anatomy, blurred, grid, cut off, draft,
watermark, signature, 
"""

height = 512
width = 256
num_inference_steps=32
# guidance_scale=15
guidance_scale=20
# guidance_scale=25
NUM = 5

pipe = StableDiffusionPipeline.from_pretrained(model_path, local_files_only=True, )

pipe = pipe.to("cpu")
pipe.safety_checker = lambda images, **kwargs: (images, False)

images = pipe(prompt, num_images_per_prompt=NUM, negative_prompt=n_prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images

with open(f"./dist/{time.time()}.txt", mode="w") as f:
  f.write(f"PROMPT: {prompt}\nN_PROMPT: {n_prompt}\nMODEL:{model_path}\nH:{height}\nW:{width}\nSTEP:{num_inference_steps}\nSCALE:{guidance_scale}")

for i, image in enumerate( images ):
  image.save(f"./dist/{time.time()}_{i}.png")