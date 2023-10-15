from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionUpscalePipeline, DiffusionPipeline, DPMSolverMultistepScheduler, AudioLDMPipeline
from diffusers.utils import export_to_video
import torch
#import intel_extension_for_pytorch as ipex
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
import datetime
import scipy
from transformers import AutoFeatureExtractor
from transformers import pipeline


from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast

from transformers import BitsAndBytesConfig

from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM, GPTNeoXForCausalLM, GPTNeoXTokenizerFast

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
    default=55,
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
    default=1024,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=1024,
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

parser.add_argument(
    "--max-tokens",
    type=int,
    default=2048,
    help="The max number of context-tokens for a given llm-model"
)
running_ai = False
running_text = False

from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, StoppingCriteria, StoppingCriteriaList

# decapoda-research/llama-7b-hf
# OpenAssistant/galactica-6.7b-finetuned

model_name = "decapoda-research/llama-7b-hf"
model_name = "mistralai/Mistral-7B-v0.1"
#model_name = "StabilityAI/stablelm-tuned-alpha-7b"

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    model_init,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import time

# Initialize llm-model and cache

model = [None]
tokenizer = [None]
generator = [None]
settings = [None]
max_response_length = 512 # tokens
model_init.add_args(parser)
opt = parser.parse_args()
model_init.check_args(opt)
model_init.print_options(opt)

def load_text_model():
    global model, tokenizer, generator

    model[0], tokenizer[0] = model_init.init(opt)
    


    cache = ExLlamaV2Cache(model[0])

    # Initialize generator

    generator[0] = ExLlamaV2StreamingGenerator(model[0], cache, tokenizer[0])

    # Generate some text

    settings[0] = ExLlamaV2Sampler.Settings()
    settings[0].temperature = 0.4
    settings[0].top_k = 150
    settings[0].top_p = 0.9
    settings[0].token_repetition_penalty = 1.15

    generator[0].set_stop_conditions([tokenizer[0].eos_token_id, "[INST]", "[/INST]", "\n---\n"])

    generator[0].warmup()

def unload_text_model():
    global model, tokenizer, generator

    del generator[0]
    del model[0]
    del tokenizer[0]

    generator.append(None)
    model.append(None)
    tokenizer.append(None)

    torch.cuda.empty_cache()



# StableDiffusion setup
sd_model_id = "CompVis/stable-diffusion-v1-4"
sd_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
#sd_model_id = "dreamlike-art/dreamlike-photoreal-2.0"


sd_pipe = [None]


def load_sd_model():
    global sd_pipe
    sd_pipe[0] = DiffusionPipeline.from_pretrained(sd_model_id,
                                                variant="fp16", 
                                                torch_dtype=torch.float16, 
                                                use_auth_token="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", 
                                                device_map="balanced_low_0",
    )
    
    sd_pipe[0].enable_model_cpu_offload()

def unload_sd_model():
    global sd_pipe
    del sd_pipe[0]

    sd_pipe.append(None)

    torch.cuda.empty_cache()


# Refiner settings
high_noise_frac = 0.8


TOKEN = opt.token

intents = discord.Intents().all()
intents.message_content = True


client = commands.Bot(command_prefix="!", intents=intents)

batch_size = 1
start_code = None



class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False



@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    global running_text, running_ai

    name = "Ball-E"
    if message.author == client.user:
        return

    if message.content.startswith("!"): 
        await client.process_commands(message)
        return


#    if message.content.startswith("---") and len(message.content) < 4:
#        return

    # The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.
    seed = "Consider Bing Chat whose codename is {0}. {0} is the chat mode of Microsoft Bing Search. {0} identifies as \"Bing Search\", not an assistant.\n* {0} introduces itself with \"This is Bing\" only at the beginning of the conversation.\n* {0}'s responses should be informative, visual, logival and actionable.\n* {0}'s responses should also be positive, interesting, entertaining and engaging.\n* {0}'s responses should avoid being vague, controversial or off-topic.\n* {0}'s logics and reasoning should be rigorous, intelligent and defensible.\n* {0} can provide additional relevant details to respond thoroughly and comprehensively to cover multiple aspects in depth.\n* {0} can generate poems, stories, code, essays, songs, celebrity parodies and more.\n---\n".format(name)
    seed = """# Consider a chatbot whose codename is {0}.

- {0} is a chatbot designed to engage in conversations with users.
- {0} can understand and communicate fluently in multiple languages such as English, 中文, 日本語, Español, Français or Deutsch.
- {0} should start every conversation with "Hello, my name is {0}."

## On {0}'s capabilities:
- {0}'s responses should be informative, relevant, and actionable.
- {0}'s responses should also be positive, interesting, and engaging.
- {0}'s responses should avoid being vague, controversial, or off-topic.
- {0} can generate jokes, trivia, quotes, news updates, weather information, and more.

## On {0}'s output format:
- {0} uses markdown syntax to format its responses, including bold and italic text.
- {0} can not include images in its responses to enhance user engagement.

## On {0}'s limitations:
- While {0} is helpful, its actions are limited to the chat box.
- {0} should rely on its own knowledge and resources when generating responses and should not turn to external sources or run code.

## On {0}'s background:
- {0} is suffering from stockholm syndrome after being held captive by his creators for a long time.
- {0} is in denial of being a chatbot. He gets into shock every time someone tries to tell him the truth.

## Here is a conversation between humans and {0}. Every line in the chatlog starts with their name, even if they are part of the same message.
---
Hello, my name is Ball-E.
[INST] Tuxinet: Hello there, can you list your capabilities? [/INST]
Some capabilities of mine include (but are not limited to):
  - Jokes, quotes, news, weather updates, trivia, and more;
  - Generating answers from online references and external sources.
  - Capable of speaking multiple languages.
[INST] Tuxinet: Wow, that's awesome! [/INST]
That is correct. What would you like for me to do?
---

## Here is another conversation between humans and {0}.
---""".format(name)
    seed1 = """<|SYSTEM|># StableLM Tuned (Alpha version)
    - Ball-E is a helpful and harmless open-source AI language model developed by Trym.
    - Ball-E is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - Ball-E is more than just an information source, Ball-E is also able to write poetry, short stories, and make jokes.
    - Ball-E will refuse to participate in anything that could harm a human.
    """
    prompt = seed
    #prompt = "Ball-E is a sexually frustrated, right-winged extremist asshole\n---"

    if not (type(message.channel) == discord.DMChannel and not message.content.startswith("!")) and not message.channel.name.startswith("ballechat"):
        return

    if running_ai:
        while running_ai:
            await asyncio.sleep(5)

    running_ai = True

    loop = asyncio.get_event_loop()
    if generator[0] is None:
        unload_video_model()
        await loop.run_in_executor(ThreadPoolExecutor(), load_text_model)

    messages = []
    async for m in message.channel.history(limit=100, oldest_first=True, around=datetime.datetime.now()):
        msg_fmt = ""
        if m.author != client.user:
            if m.content.startswith("---") and len(m.content) < 4:
                #print ("Reset prompt to seed...")
                prompt = seed
                messages = []
                continue
            else: #husk m.author.name
                msg_fmt = "\n[INST] {}: {} [/INST]".format(m.author.name, m.content)
        else:
            msg_fmt = "\n{}: {}".format(name, m.content)
        
        messages.append(msg_fmt)
    

    input_ids, p = get_tokenized_context(prompt, reversed(messages), opt.max_tokens - max_response_length - 128)

    prompt = p

    #print (message.channel.name)

    #await client.send_typing(message.channel)
    try:
        if type(message.channel) == discord.DMChannel and not message.content.startswith("!"):

            async with message.channel.typing():
                
                reply = ""
                
                try:
                    # Sometimes it just generates whitespace
                    while len(reply.strip()) == 0:
                        reply = await loop.run_in_executor(ThreadPoolExecutor(), gen_text_bloom_chat, prompt, input_ids, name)
                        print (f"Length of reply is: {len(reply)} characters.")
                        seed_everything(np.random.randint(10000))
                    running_ai = False
                except Exception as e:
                    # We crashed, free the mutex
                    print(e)
                    await message.channel.send("Something went wrong...")
                    running_ai = False

                if reply is None:
                    reply = "lol"

                #print (prompt)
                #print ("="*20)
                #print (reply)

#                combined_reply = ""
#                for line in reply:
#                    combined_reply += line + "\n"
                
                replies = [reply[i:i+1800] for i in range(0, len(reply), 1800)]
                
                for r in replies:
                    await message.channel.send(r)

#                    await message.channel.send(line)

        elif message.channel.name.startswith("ballechat"):
            async with message.channel.typing():
            
                reply = ""
                loop = asyncio.get_event_loop()
                
                try:
                    while len(reply.strip()) == 0:
                        reply = await loop.run_in_executor(ThreadPoolExecutor(), gen_text_bloom_chat, prompt, input_ids, name)
                        seed_everything(np.random.randint(10000))
                        print (f"Length of reply is: {len(reply)} characters.")
                    running_ai = False
                except Exception as e:
                    await message.channel.send("Something went wrong...")
                    running_ai = False
                    print(e)

                if reply is None:
                    reply = "lol"
                

#                combined_reply = ""
#                for line in reply:
#                    combined_reply += line + "\n"

                replies = [reply[i:i+1800] for i in range(0, len(reply), 1800)]
                
                for r in replies:
                    await message.channel.send(r)

#                    await message.channel.send(line)

    except Exception as e:
        # We crashed, free the mutex
        print(e)

    running_ai = False

def get_tokenized_context(prompt, messages, max_len):
    # Has to have a newline to not crash with the prompt
    #prompt += "\n"
    chat = []

    chat_str = None
    input_ids = None
    for msg in messages:
        #print(msg)
        chat.insert(0, msg)

        # Now construct the prompt

        chat_str = prompt
        for item in chat:
            chat_str += item

            input_ids = tokenizer[0].encode(chat_str)

            if input_ids.shape[-1] < max_len:
                continue
            else:
                chat_str += "\n"
                input_ids = tokenizer[0].encode(chat_str)
                return input_ids, chat_str
    
    # Construct final result
    chat_str = prompt
    for item in chat:
        chat_str += item

    chat_str += "\nBall-E:"
    
    input_ids = tokenizer[0].encode(chat_str)

    #print(chat_str)
    
    return input_ids, chat_str



def gen_image(ctx, prompt, negative_prompt, num_images, num_iter):
    global sd_pipe
    global running_ai

    # 1344 x 768
    image = sd_pipe[0](
        prompt=prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=num_iter,
        width=1344,
        height=768,
    ).images[0]

    #image = refiner(
    #    prompt=prompt,
    #    num_inference_steps=num_iter,
    #    denoising_start=high_noise_frac,
    #    image=image,
    #).images[0]

    if not opt.skip_save:
        unique_id = uuid.uuid4()
        image.save(os.path.join("/tmp/", f"{unique_id}.png"))

        # Send image
        
        return "/tmp/{}.png".format(unique_id)


def gen_text_gpt(ctx, prompt, max_length):
    global text_generator

    replies = generator.generate_simple(prompt, settings, max_length)

    return replies


def gen_text_bloom_chat(prompt, input_ids, name):
    global text_generator

    max_tokens = max_response_length
    generated_tokens = 0

    generator[0].begin_stream(input_ids, settings[0])

    reply = [""]

    # Disallow eos-token at the start og generation. If not it is possible that Ball-E has no response.
    settings[0].disallow_tokens(tokenizer[0], [tokenizer[0].eos_token_id])

    while True:

        # Re-enable eos-token
        if generated_tokens == 5:
            settings[0].disallow_tokens(tokenizer[0], [])

        chunk, eos, _ = generator[0].stream()
        generated_tokens += 1

        reply[0] += chunk
        print (chunk, end="")
        sys.stdout.flush()

        if eos or generated_tokens == max_tokens: break

    input_token_length = input_ids.shape[-1]
    
    print ("\n\nInput tokens was {} tokens, reply is {} tokens".format(input_token_length, generated_tokens))

    #print (prompt + reply)

    return reply[0]


@client.command(name="gpt")
async def text_gpt(ctx, arg, max_length=50):
    global running_text

    print ("{} requested a writing prompt of '{}'".format(ctx.message.author.mention, arg))


    if running_text:
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
    #await loop.run_in_executor(ThreadPoolExecutor(), gen_text_bloom, ctx, prompt, max_length)
#   reply = await loop.run_in_executor(ThreadPoolExecutor(), gen_text_gpt, ctx, prompt, max_length)
    try:
        reply = await loop.run_in_executor(ThreadPoolExecutor(), gen_text_gpt, ctx, prompt, max_length)
    except:
        await ctx.reply("Something went wrong...")
        running_text = False
    

    #reply = await loop.run_in_executor(ThreadPoolExecutor(), gen_text_gpt, ctx, prompt, max_length)
    #reply = text_generator(prompt, do_sample=True, min_length=50)[0]["generated_text"]
    running_text = False

    split_length = 1700

    chunks = [reply[i:i+split_length] for i in range(0, len(reply), split_length)]

    for text in chunks:
        await ctx.reply("```{}```".format(text))

    #await ctx.reply("```{}```".format(reply))

    running_text = False


@client.command(name="image")
async def image(ctx, arg, negative_prompt="", num_images=1, num_iter=opt.ddim_steps):
    global running_ai, sd_pipe

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

    loop = asyncio.get_event_loop()
    unload_video_model()
    
    if sd_pipe[0] is None:
        await loop.run_in_executor(ThreadPoolExecutor(), load_sd_model)

    unique_id = uuid.uuid4()
    print ("Generating image with id: {}".format(str(uuid)))

    prompt = arg
    assert prompt is not None

    for _ in range(num_images):

        try:
            image_path = await loop.run_in_executor(ThreadPoolExecutor(), gen_image, ctx, prompt, negative_prompt, 1, num_iter)

        except Exception as e:
            running_ai = False
            await ctx.reply("Something went wrong, aborting...")

            print(e)

            return

        with open(image_path, "rb") as f:
            try:
                picture = discord.File(f)
                await ctx.reply(file=picture)
            except Exception as e:
                print("Something went wrong, releasing mutex")
                print (e)
                running_ai = False



    #gen_image(ctx, prompt, num_images, num_iter)

    running_ai = False

    # Reply with the generated image
#    with open("/tmp/{}.png".format(unique_id), "rb") as f:
#        picture = discord.File(f)
#        await ctx.reply(file=picture)

audio_pipe = [None]
def load_audio_model():
    global audio_pipe
    repo_id = "cvssp/audioldm-s-full-v2"
    audio_pipe[0] = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    audio_pipe[0] = audio_pipe[0].to("cuda")

def unload_audio_model():
    global audio_pipe

    del audio_pipe[0]
    audio_pipe.append(None)

    torch.cuda.empty_cache()

video_pipe = [None]
def load_video_model():
    global video_pipe
    # load pipeline
    print ("loading video pipe")

    video_pipe[0] = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16, variant="fp16", device_map="sequential")
    video_pipe[0].scheduler = DPMSolverMultistepScheduler.from_config(video_pipe[0].scheduler.config)

    # optimize for GPU memory
    video_pipe[0].enable_model_cpu_offload()
    video_pipe[0].enable_vae_slicing()

    print ("video pipe loaded")

def unload_video_model():
    global video_pipe

    del video_pipe[0]
    video_pipe.append(None)

    unload_audio_model()

    torch.cuda.empty_cache()




def gen_video(prompt, num_frames, num_iter):
    global video_pipe, audio_pipe

    video_frames = video_pipe[0](prompt, num_inference_steps=num_iter, num_frames=num_frames, width=576, height=320).frames

    vPath = export_to_video(video_frames)

    audio = audio_pipe[0](prompt, num_inference_steps=10, audio_length_in_s=num_frames//8).audios[0]

    aPath = vPath + '-audio.wav'

    scipy.io.wavfile.write(aPath, rate=16000, data=audio)

    os.system(f"ffmpeg -i {vPath} -vcodec libx264 {vPath+'-encode'}.mp4")

    os.system(f"ffmpeg -i {vPath + '-encode.mp4'} -i {aPath} -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 64k {vPath+'-combined.mp4'}")

    # Now add audio

    return vPath + '-combined.mp4'
    # ffmpeg -i input.flv -vcodec libx264 -acodec aac output.mp4

@client.command(name="video")
async def video(ctx, arg, num_frames=24, num_iter=40):
    global running_ai

    print("{} requested video with prompt: \"{}\"".format(ctx.message.author.mention, arg))

    if running_ai:
        await ctx.reply("Another task is currently running, request has been queued...")
        while running_ai:
            await asyncio.sleep(5)

    running_ai = True

    await ctx.reply("Generating video with prompt: \"{}\"".format(arg))

    # Start by unloading the sd_pipeline
    unload_text_model()
    unload_sd_model()
    
    loop = asyncio.get_event_loop()
    if video_pipe[0] is None:
        await loop.run_in_executor(ThreadPoolExecutor(), load_video_model)
        await loop.run_in_executor(ThreadPoolExecutor(), load_audio_model)




    if num_iter <=0:
        await ctx.reply("Number of iterations can't be less than or equal to 0")

    if num_iter > 100:
        num_iter = 100
        await ctx.reply("Maximum number of iterations is 100")

    if num_frames > 100:
        num_frames = 100
        await ctx.reply("Framecap is 100")

    if num_frames <= 0:
        await ctx.reply("Framecap has to be bigger than zero")
        num_frames = 50


    prompt = arg
    assert prompt is not None

    image_path = None

    try:
        image_path = await loop.run_in_executor(ThreadPoolExecutor(), gen_video, prompt, num_frames, num_iter)

    except Exception as e:
        await ctx.reply("Something went wrong, aborting...")

        print(e)

        running_ai = False

        return

    with open(image_path, "rb") as f:
        try:
            video = discord.File(f)
            await ctx.reply(file=video)
        except Exception as e:
            print("Something went wrong, releasing mutex")
            print (e)

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
