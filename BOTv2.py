from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionUpscalePipeline
import torch


import argparse, os, sys, glob
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from transformers import AutoFeatureExtractor
from transformers import pipeline


from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast

import discord
from discord.ext import commands
import asyncio

from concurrent.futures import ThreadPoolExecutor

import uuid


TOKEN = ''
client = None

parser = argparse.ArgumentParser()

parser.add_argument(
    "--token",
    type=str,
    nargs="?",
    default="",
    help="The token for the bot on discord"
)

parser.add_argument(
        "--fp16",
        action='store_true',
        help="before running inference, convert the model to fp16 "
             "(this saves runtime memory, though at a potential loss of some quality)",
        default=False
    )

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples"
)
parser.add_argument(
    "--skip_grid",
    action='store_true',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=45,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--laion400m",
    action='store_true',
    help="uses the LAION400M model",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=3,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=9,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="models/ldm/stable-diffusion-v1/model.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=np.random.randint(10000),
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)

running_ai = False
running_text = False


opt = parser.parse_args()


sd_model_id = "stabilityai/stable-diffusion-2-base"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(sd_model_id, subfolder="scheduler")
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16, use_auth_token="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")
#sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, scheduler=scheduler)
sd_pipe = sd_pipe.to("cuda")

#text_generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
#text_generator = pipeline('text-generation', model='bigscience/bloom-1b3')

model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b3")
tokenisex = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b3")

#prompt = "hanelo"
#image = sd_pipe(prompt).images[0]

#del sd_pipe

#sd_pipe.enable_attention_slicing()


upscaler_model_id = "stabilityai/stable-diffusion-x4-upscaler"
#up_pipe = StableDiffusionUpscalePipeline.from_pretrained(upscaler_model_id, revision="fp16", torch_dtype=torch.float16)
#up_pipe = up_pipe.to("cuda")

#up_pipe.enable_attention_slicing()

TOKEN = opt.token

intents = discord.Intents().all()
intents.message_content = True


client = commands.Bot(command_prefix="!", intents=intents)

batch_size = 1
start_code = None



@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

#@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('!hello'):
        await message.channel.send('Hello!')


def gen_image(ctx, prompt, num_images, num_iter):
    global sd_pipe
    global running_ai
    data = [batch_size * [prompt]]


    for n in trange(num_images, desc="Sampling"):
        for prompt in tqdm(data, desc="data"):
            #sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
            #sd_pipe = sd_pipe.to("cuda")
            image = sd_pipe(prompt).images[0]

            #del sd_pipe

            #up_pipe = StableDiffusionUpscalePipeline.from_pretrained(upscaler_model_id, revision="fp16", torch_dtype=torch.float16)
            #up_pipe = up_pipe.to("cuda")
            #upscaled_image = up_pipe(prompt=prompt, image=image).images[0]  

            #del up_pipe

            if not opt.skip_save:
                unique_id = uuid.uuid4()
                image.save(os.path.join("/tmp/", f"{unique_id}.png"))

                # Send image
                
                return "/tmp/{}.png".format(unique_id)

                #base_count += 1


    try:

        for n in trange(num_images, desc="Sampling"):
            for prompt in tqdm(data, desc="data"):
                image = sd_pipe(prompt).images[0]

                upscaled_image = up_pipe(prompt=prompt, image=image).images[0]  

                if not opt.skip_save:
                    unique_id = uuid.uuid4()
                    upscaled_image.save(os.path.join("/tmp/", f"{unique_id}.png"))

                    # Send image
                    
                    return "/tmp/{}.png".format(unique_id)

                    #base_count += 1
    except:
        # We crashed, free the mutex
        running_ai = False


def gen_text_gpt(ctx, prompt, max_length):
    global text_generator

    return text_generator(prompt, do_sample=True, max_length=max_length)[0]["generated_text"]


@client.command(name="text")
async def text(ctx, arg, max_length=100):
    global running_text

    print ("{} requested a writing prompt of '{}'".format(ctx.message.author.mention, arg))


    if running_text and False:
        await ctx.reply("Another task is currently running, request has been queued...")
        while running_text:
            await asyncio.sleep(5)

    running_text = True

    await ctx.reply("Generating text, this might take some time...")

    if max_length > 2000:
        await ctx.reply("Maximum number of tokens has to be a number less than 2000")
        max_length = 2000
    
    prompt = arg
    assert prompt is not None

    reply = None
    loop = asyncio.get_event_loop()
    
    try:
        reply = await loop.run_in_executor(ThreadPoolExecutor(), gen_text_gpt, ctx, prompt, max_length)
    except:
        await ctx.reply("Something went wrong...")
        running_text = False
    

    #reply = await loop.run_in_executor(ThreadPoolExecutor(), gen_text_gpt, ctx, prompt, max_length)
    #reply = text_generator(prompt, do_sample=True, min_length=50)[0]["generated_text"]

    await ctx.reply("```{}```".format(reply))

    running_text = False




@client.command(name="image")
async def image(ctx, arg, num_images=1, num_iter=opt.ddim_steps):
    global running_ai

    print("{} requested {} image(s) with prompt: \"{}\"".format(ctx.message.author.mention, num_images, arg))

    if running_ai:
        await ctx.reply("Another task is currently running, request has been queued...")
        while running_ai:
            await asyncio.sleep(5)

    running_ai = True


    await ctx.reply("Generating image with prompt: \"{}\"".format(arg))

    if num_images <= 0:
        await ctx.reply("Number of images needs to be a positive number (5 max)")
        return

    if num_images > 5:
        num_images = 5
        await ctx.reply("num_images threshold is 5")

    if num_iter > 100:
        num_iter = 100
        await ctx.reply("Maximum number of iterations is 100")


    unique_id = uuid.uuid4()
    print ("Generating image with id: {}".format(uuid))

    prompt = arg
    assert prompt is not None
    loop = asyncio.get_event_loop()

    for _ in range(num_images):

        image_path = await loop.run_in_executor(ThreadPoolExecutor(), gen_image, ctx, prompt, 1, num_iter)

        try:
            print ("lol")
            #image_path = await loop.run_in_executor(ThreadPoolExecutor(), gen_image, ctx, prompt, 1, num_iter)

        except:
            running_ai = False
            await ctx.reply("Something went wrong, aborting...")
            return

        with open(image_path, "rb") as f:
            try:
                picture = discord.File(f)
                await ctx.reply(file=picture)
            except:
                print("Something went wrong, releasing mutex")
                running_ai = False



    #gen_image(ctx, prompt, num_images, num_iter)

    running_ai = False

    # Reply with the generated image
#    with open("/tmp/{}.png".format(unique_id), "rb") as f:
#        picture = discord.File(f)
#        await ctx.reply(file=picture)


def main():

    print("Model loaded")

    seed_everything(np.random.randint(10000))



    client.run(TOKEN)


if __name__ == '__main__':
    main()