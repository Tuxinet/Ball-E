#from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionUpscalePipeline
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
model_name = "meta-llama/Llama-2-7b-hf"
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

# Initialize model and cache

model_init.add_args(parser)
opt = parser.parse_args()
model_init.check_args(opt)
model_init.print_options(opt)
model, tokenizer = model_init.init(opt)

model_directory =  "/home/tuxinet/git/Llama2-13B-4.0bpw-h6-exl2/"
model_directory = "/home/tuxinet/models/chimera-inst-chat-13b-gptq-4bit/"
model_directory = "/home/tuxinet/models/Wizard-Vicuna-13B-Uncensored-GPTQ/"
model_directory = "/home/tuxinet/models/CodeLlama-13B-instruct-2.65bpw-h6-exl2/"

max_response_length = 512 # tokens

cache = ExLlamaV2Cache(model)

# Initialize generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

# Generate some text

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.15
#settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

generator.set_stop_conditions(["[INST]", "[/INST]", "[REPLY]", "[/REPLY]", tokenizer.eos_token_id])
#generator.set_stop_conditions(["[INST]"])

generator.warmup()

#device_map['model.decoder.layers.9'] = "cpu"
#device_map['model.decoder.layers.10'] = "cpu"
#device_map['model.decoder.layers.11'] = "cpu"

#quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)


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
    global running_text

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
            msg_fmt = "\n{}".format(m.content)
        
        messages.append(msg_fmt)
    

    input_ids, p = get_tokenized_context(prompt, reversed(messages), opt.max_tokens - max_response_length - 128)

    prompt = p


    if running_text:
        while running_text:
            await asyncio.sleep(5)

    running_text = True

    #print (message.channel.name)

    #await client.send_typing(message.channel)
    try:
        if type(message.channel) == discord.DMChannel and not message.content.startswith("!"):

            async with message.channel.typing():
                
                reply = ""
                loop = asyncio.get_event_loop()
                
                try:
                    # Sometimes it just generates whitespace
                    while len(reply.strip()) == 0:
                        reply = await loop.run_in_executor(ThreadPoolExecutor(), gen_text_bloom_chat, prompt, input_ids, name)
                        print (f"Length of reply is: {len(reply)} characters.")
                        seed_everything(np.random.randint(10000))
                    running_text = False
                except Exception as e:
                    # We crashed, free the mutex
                    print(e)
                    await message.channel.send("Something went wrong...")
                    running_text = False

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
                    running_text = False
                except Exception as e:
                    await message.channel.send("Something went wrong...")
                    running_text = False
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

    running_text = False

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

            input_ids = tokenizer.encode(chat_str)

            if input_ids.shape[-1] < max_len:
                continue
            else:
                chat_str += "\n"
                input_ids = tokenizer.encode(chat_str)
                return input_ids, chat_str
    
    # Construct final result
    chat_str = prompt
    for item in chat:
        chat_str += item

    chat_str += "\n"
    
    input_ids = tokenizer.encode(chat_str)

    #print(chat_str)
    
    return input_ids, chat_str



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
    except Exception as e:
        # We crashed, free the mutex
        print(e)
        running_ai = False


def gen_text_gpt(ctx, prompt, max_length):
    global text_generator

    replies = generator.generate_simple(prompt, settings, max_length)

    return replies


def gen_text_bloom_chat(prompt, input_ids, name):
    global text_generator
    #print (prompt)

    num_lines_prompt = len(prompt.split("\n")) - 1  # Prompt ends with Ball-E:, so we have to do a minus one to include if balle just does one answer
    
    # We want to keep on generating until the last line starts with [INST]
    current_pointer = num_lines_prompt + 2

    max_tokens = max_response_length
    tokens_per_iter = 5
    replies = None
    last_replies = None
    generated_tokens = 0


    generator.begin_stream(input_ids, settings)

    reply = [""]

    # Disallow eos-token at the start og generation. If not it is possible that Ball-E has no response.
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    while True:

        # Re-enable eos-token
        if generated_tokens == 5:
            settings.disallow_tokens(tokenizer, [])

        chunk, eos, _ = generator.stream()
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
