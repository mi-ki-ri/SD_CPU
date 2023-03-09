import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import time

model_path = "./models/trin"

prompt = """
A snapshot of ((dancing)) (girl), ((contraposto)), in the rain, low angle, full body, sketch by color pencil, art by akihiko yoshida
"""

n_prompt = """
2girls, ugly, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, blurred, watermark, grainy, signature, cut off, draft
"""

height = 512
width = 512
num_inference_steps=50
# guidance_scale=15
guidance_scale=20
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