import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import discord
from discord.ext import commands
import asyncio

from concurrent.futures import ThreadPoolExecutor

import uuid


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x




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

# Settings up the deep-learning shit
config = OmegaConf.load("v1-inference.yaml")
model = load_model_from_config(config, "model.ckpt")

model = model.half()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

#sampler = PLMSSampler(model)
sampler = DDIMSampler(model)


opt = parser.parse_args()

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
    global running_ai
    data = [batch_size * [prompt]]


    try:
        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"), \
            model.ema_scope():
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(num_images, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=num_iter,
                                                            conditioning=c,
                                                            batch_size=1,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            #x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                            x_checked_image = x_samples_ddim

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            if not opt.skip_save:
                                for x_sample in x_checked_image_torch:
                                    unique_id = uuid.uuid4()
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img.save(os.path.join("/tmp/", f"{unique_id}.png"))

                                    # Send image
                                    
                                    return "/tmp/{}.png".format(unique_id)

                                    #base_count += 1

                            if not opt.skip_grid:
                                all_samples.append(x_checked_image_torch)
    except:
        # We crashed, free the mutex
        running_ai = False


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

        try:

            image_path = await loop.run_in_executor(ThreadPoolExecutor(), gen_image, ctx, prompt, 1, num_iter)

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
