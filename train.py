import os
import pickle
import requests
import numpy as np
import random
from tqdm import tqdm
import copy
import time

from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import wandb
import torch.nn.functional as F
import math

from model import GPTConfig, GPT
from main_utilities import *
from evaluation import *

def create_meta_for_addition(data):
    """Create metadata for addition data."""
    # Define the vocabulary for addition problems
    # This includes digits, operators, equals sign, and newline
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    # Create encoder and decoder dictionaries
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    meta = {
        'vocab_size': vocab_size,
        'vocab': chars,
        'stoi': stoi,
        'itos': itos
    }
    return meta

def encode_addition(text, meta):
    """Encode text to tensor using the metadata."""
    return torch.tensor([meta['stoi'][c] for c in text], dtype=torch.long)

def decode_addition(tensor, meta):
    """Decode tensor to text using the metadata."""
    if isinstance(tensor, torch.Tensor):
        return ''.join([meta['itos'][i.item()] for i in tensor])
    else:
        return ''.join([meta['itos'][i] for i in tensor])

class AdditionDataset(Dataset):
    def __init__(self, file_path, meta):
        self.meta = meta
        # Read the text file
        with open(file_path, 'r') as f:
            self.lines = f.readlines()
        # Remove any empty lines and strip whitespace
        self.lines = [line.strip() for line in self.lines if line.strip()]
        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line = self.lines[idx]
        # Convert the line to tensor using our encoder
        x = encode_addition(line[:-1], self.meta)  # all but last char
        y = encode_addition(line[1:], self.meta)   # all but first char
        return x, y

# I/O

out_dir = '/drive/MyDrive/addition/plain_no_pad/out'
resume_dir = None
resume_iter = False # if True, resume from saved iter_num, otherwise resume from iter_num 0
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_entity = 'ssdd'
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
exp_name = 'default_exp_name'

# data
dataset = 'bal'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
test_batch_size = 128
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
train_data_path = 'train.bin'
val_data_path = 'val.bin'
multi_digit = False
num_digit = 3
binary = False

# using two data - data1 = text / data2 = addition
train_both = False # use seperate text/add data for train/val (get_batch uses this to sample from two differernt datasets)
data_ratio = 0.2 # ratio of data_path2 compared with data_path1
train_data_path2 = 'train_addition.bin' # only used when train_both = True
val_data_path2 = 'val_addition.bin'

# evaluation
eval_text = False # if True get perplexity using eval_text_data_path
eval_text_data_path = None # directory to text data (.bin file) - ex. 'data/shakespeare_add_ar_mixed/val_text.bin'
eval_addition = False # if True compute test accuracy of "a+b="
start = None
eval_addition_ar = False
start_ar = None
eval_other = False # use this to evaluate other operations (ex. train on operator '-' but evaluate on other_operator '+')
start_other = None
other_operator = '+'
eval_addition_train = False
start_train = None
reverse_ab = False
reverse_c = False
zero_pad = False
algo_reason = False
add_space = False
analysis = False
num_addition = 4

# model
n_layer = 6
n_head = 6
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
ckpt_path_name = 'ckpt.pt'
save_final = True

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = None # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
use_flash = True
data_type = 'binary' # 'binary' by default, can be 'text'
operator = '+' # can be '+', '-', '*', 'sin', 'sqrt'
data_shuffle = True
data_format = 'plain' # 'plain' or 'reverse' or 'algo_reasoning'
vocabulary = 'all_ascii_chars' # can be 'all_ascii_chars' or 'numbers_only' or 'custom_input_data'
meta_path_specified = True # use saved meta_file (False if data_type='text')
eps = 0
tokenizer = 'char' # by default, use char level tokenizer. but for pretrained models, use openai tokenizer eg: 'gpt2'

simple=False
random_A=False
random_C=False

use_lora = False # use lora (from minLoRA)
print_interval = 2  # if we're using gpt-2 model, I want to see it prompted on text


config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# function to set seed for all random number generators
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # to make sure GPU runs are deterministic even if they are slower set this to True
    torch.backends.cudnn.deterministic = False
    # warning: this causes the code to vary across runs
    torch.backends.cudnn.benchmark = True
    print("Seeded everything: {}".format(seed))

if min_lr == None:
    min_lr = learning_rate/10
master_process = True
seed_offset = 0
if master_process:
  os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cudnn.benchmark = True # cudnn auto-tuner
torch.backends.cudnn.deterministic = False # cudnn auto-tuner
# this is probably overkill but seed everything again
set_seed(1337 + seed_offset)

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Read the data files
with open(train_data_path, 'r') as f:
    train_data = f.read()
with open(val_data_path, 'r') as f:
    val_data = f.read()

# Create metadata from the combined data
all_data = train_data + val_data
meta = create_meta_for_addition(train_data)
meta_vocab_size = meta['vocab_size']
print(f"Using vocabulary size: {meta_vocab_size}")


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
best_perplexity = 1e9 # on text data
best_accuracy = -1 # on addition data


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, use_flash=use_flash) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)
elif init_from == 'resume':
    if resume_dir:
        print(f"Resuming training from {resume_dir}")
        checkpoint = torch.load(resume_dir, map_location=device)
    else:
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, ckpt_path_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num'] if resume_iter else 0
    max_iters += iter_num
    best_val_loss = checkpoint['best_val_loss']
    if 'best_perplexity' in checkpoint.keys():
        best_perplexity = checkpoint['best_perplexity']
    if 'best_accuracy' in checkpoint.keys():
        best_accuracy = checkpoint['best_accuracy']

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # Get an iterator from the DataLoader
        dataloader = train_loader if split == 'train' else val_loader
        dataloader_iter = iter(dataloader)

        for k in range(eval_iters):
            try:
                X, Y = next(dataloader_iter)

            except StopIteration:
                # If we run out of batches, create a new iterator
                dataloader_iter = iter(dataloader)
                X, Y = next(dataloader_iter)

            with ctx:
                X, Y = X.to(device), Y.to(device)
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr_for_iter(iter_num):
    """Calculate learning rate based on iteration number using cosine decay with warmup."""
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / warmup_iters
    
    if iter_num >= lr_decay_iters:
        return min_lr
    
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, dir = out_dir)




train_dataset = AdditionDataset(train_data_path, meta)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device_type=='cuda')
)

val_dataset = AdditionDataset(val_data_path, meta)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device_type=='cuda')
)

# encode, decode = get_encode_decode(meta_path, tokenizer=tokenizer)

result_dict = {'iter' : [], 'train_loss': [], 'val_loss': [], 'test_acc': []}

result_dir = get_results_dir(config)
config['result_dir'] = result_dir
with open(os.path.join(result_dir, "config.yaml"), "w") as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

import time
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0
iter_num = 0

max_iters = config.get('max_iters', 10000)
 # number of epochs to warm up learning rate

# Initialize tracking variables
iter_num = 0
best_val_loss = 1e9
best_accuracy = -1
running_mfu = -1.0

# Create infinite data loader
def get_infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

train_loader_iter = get_infinite_dataloader(train_loader)
if 'max_new_tokens' in config.keys():
    print(f"max_new_tokens: {config['max_new_tokens']}")
else:
    print(f"max_new_tokens used: {num_digit+2}")

# Training loop - iteration based
while iter_num < max_iters:
    model.train()
    
    # Get learning rate for current iteration
    if decay_lr:
        lr = get_lr_for_iter(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Get next batch
    X, Y = next(train_loader_iter)
    X, Y = X.to(device), Y.to(device)
    
    # Forward pass
    with ctx:
        logits, loss = model(X, Y)
    
    # Backward pass
    scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    
    # Evaluation
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
        if eval_addition:
            config['start'] = start
            test_accuracy, _ , correct, incorrect = evaluate_addition_batch(config, model, ctx, encode=lambda x: encode_addition(x, meta),
                decode=lambda x: decode_addition(x, meta), verbose=False, num_digit=num_digit, zero_pad=zero_pad,
                                                    reverse_ab=reverse_ab, reverse_c=reverse_c,
                                                    data_type=data_type, operator=operator, data_format=data_format, analyze=True)
            
            if test_accuracy > best_accuracy and iter_num % 5 * eval_interval == 0:
                best_accuracy = test_accuracy
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'best_accuracy': best_accuracy,
                    'config': config,
                    'meta': meta,
                }
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_iter_{iter_num}_acc.pt'))
        
        result_dict['iter'].append(iter_num)
        result_dict['train_loss'].append(losses['train'].item())
        result_dict['val_loss'].append(losses['val'].item())
        result_dict['test_acc'].append(test_accuracy if eval_addition else None)  # We don't have train accuracy during iterations
        
        # Save results to CSV after each evaluation
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "test/accuracy": test_accuracy if eval_addition else None,
            })
    
    iter_num += 1

# Save final checkpoint
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'best_accuracy': best_accuracy,
    'config': config,
    'meta': meta,
}
torch.save(checkpoint, os.path.join(out_dir, f'ckpt_final.pt'))


losses = estimate_loss()

if eval_addition:
  config['start'] = start
  test_accuracy, _ , correct, incorrect = evaluate_addition_batch(config, model, ctx, encode=lambda x: encode_addition(x, meta),
                decode=lambda x: decode_addition(x, meta), verbose=False, num_digit=num_digit, zero_pad=zero_pad,
                                                    reverse_ab=reverse_ab, reverse_c=reverse_c,
                                                    data_type=data_type, operator=operator, data_format=data_format, analyze=True)
  import csv
  # Save correct examples
  correct_path = os.path.join(result_dir, 'correct_examples.csv')
  with open(correct_path, 'w', newline='') as csvfile:
    fieldnames = ['operands', 'result', 'outcome', 'c_hat2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, nums in enumerate(correct):
        operands, result, outcome, c_hat2 = nums
        writer.writerow({'operands': operands, 'result': result, 'outcome': outcome, 'c_hat2': c_hat2})
  
  # Save incorrect examples
  incorrect_path = os.path.join(result_dir, 'incorrect_examples.csv')
  with open(incorrect_path, 'w', newline='') as csvfile:
    fieldnames = ['operands', 'result', 'outcome', 'c_hat2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, nums in enumerate(incorrect):
        operands, result, outcome, c_hat2 = nums
        writer.writerow({'operands': operands, 'result': result, 'outcome': outcome, 'c_hat2': c_hat2})
if eval_addition_train:
    config['start'] = start_train
    train_accuracy, *_ = evaluate_addition_batch(config, model, ctx, encode=lambda x: encode_addition(x, meta),
                decode=lambda x: decode_addition(x, meta), verbose=False, num_digit=num_digit, zero_pad=zero_pad,
                                                    reverse_ab=reverse_ab, reverse_c=reverse_c,
                                                    data_type=data_type, operator=operator, data_format=data_format)

if wandb_log:
  wandb_dict = {
      "iter": iter_num,
      "train/loss": losses['train'],
      "val/loss": losses['val'],
      "lr": lr,
      "test/accuracy": test_accuracy if eval_addition else None,
  }
  wandb.log(wandb_dict)

result_dict['iter'].append(iter_num)
result_dict['train_loss'].append(losses['train'].item())
result_dict['val_loss'].append(losses['val'].item())
result_dict['test_acc'].append(test_accuracy if eval_addition else None)

result_df = pd.DataFrame(result_dict)
result_df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'best_accuracy': best_accuracy,
    'config': config,
}

if wandb_log:
    wandb.finish()
