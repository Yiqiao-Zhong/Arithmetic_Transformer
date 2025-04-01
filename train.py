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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
import wandb
import torch.nn.functional as F
import math

from model import GPTConfig, GPT
from main_utilities import *
from evaluation import *


class AdditionDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = torch.from_numpy((self.data[idx:idx+self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[idx+1:idx+1+self.block_size]).astype(np.int64))
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

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
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

if data_type == 'binary':
  data_dir = os.path.join('data', dataset)
  train_data = np.memmap(os.path.join(data_dir, train_data_path), dtype=np.uint16, mode='r')
  val_data = np.memmap(os.path.join(data_dir, val_data_path), dtype=np.uint16, mode='r')
  if train_both:
      train_data2 = np.memmap(os.path.join(data_dir, train_data_path2), dtype=np.uint16, mode='r')
      val_data2 = np.memmap(os.path.join(data_dir, val_data_path2), dtype=np.uint16, mode='r')
  if eval_text:
      if eval_text_data_path is None:
          print('eval_text_data_path is None!!! No binary file to evaluate perplexity on.')
      eval_text_data = np.memmap(eval_text_data_path, dtype=np.uint16, mode='r')
  # test_data_str = None # test_data for addition testing will be handled with "start"
  meta_path = None
else:
    if data_type == 'text':
      if ('reverse' in data_format and not reverse_c) or (reverse_c and 'reverse' not in data_format):
          raise ValueError('reverse_c must be True for data_format == "reverse"')
      elif (data_format == 'algo_reasoning' and not algo_reason) or (algo_reason and data_format != 'algo_reasoning'):
          raise ValueError('algo_reason must be True for data_format == "algo_reasoning"')

    meta_path_specified = False

    data_dir = os.path.join('data', dataset)
    train_data_path = os.path.join(data_dir, train_data_path)
    # val_data = os.path.join(data_dir, val_data_path)
    train_data_list = get_data_list(train_data_path, operator=operator)
    val_data_list = get_data_list(filename=None, operator=operator) # get_data_list(val_data, operator='+')
    train_data_str = generate_data_str(train_data_list, operator=operator, format=data_format, train=True, shuffle=data_shuffle, add_space=add_space, simple=simple, random_A=random_A, random_C=random_C)
    val_data_str = generate_data_str(val_data_list, operator=operator, format=data_format, train=True, shuffle=data_shuffle, add_space=add_space, simple=simple, random_A=random_A, random_C=random_C)
    meta, meta_path, data_encoder, data_decoder = create_meta_file(vocabulary=vocabulary, input_data_str=train_data_str, tokenizer=tokenizer)
    meta_vocab_size = meta['vocab_size']
    train_data = data_encoder(train_data_str)
    val_data = data_encoder(val_data_str)
    if eval_addition_train and start_train is None:
        # specify the start_train to be oour train data file
        start_train = f"FILE:{train_data_path}"

    if train_both:
        # This is for the case where we use two different datasets for training
        # we sample from both with a specified ratio - data_ratio
        # TODO: let's leave this here for now.
        train_data2 = np.memmap(os.path.join(data_dir, train_data_path2), dtype=np.uint16, mode='r')
        val_data2 = np.memmap(os.path.join(data_dir, val_data_path2), dtype=np.uint16, mode='r')

    if eval_text:
        # eval_text_data = np.memmap(eval_text_data_path, dtype=np.uint16, mode='r')
        text_data_list = get_data_list(eval_text_data_path, operator='text')
        text_data_str = generate_data_str(text_data_list, operator='text', format=data_format, train=False, shuffle=False)
        eval_text_data = data_encoder(text_data_str)


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
best_perplexity = 1e9 # on text data
best_accuracy = -1 # on addition data


if meta_path_specified:
    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    else:
        meta_path = None

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

# if block_size < model.config.block_size:
#     model.crop_block_size(block_size)
#     model_args['block_size'] = block_size # so that the checkpoint will have the right value
# initialize a GradScaler. If enabled=False scaler is a no-op
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

def get_lr_for_epoch(epoch):
    """
    Calculate learning rate based on epoch number using cosine decay with warmup.

    Args:
        epoch (int): Current epoch number (0-indexed)
        config (dict): Configuration containing learning rate parameters

    Returns:
        float: Learning rate for the current epoch
    """


    if epoch < warmup_epochs:
        return learning_rate * (epoch + 1) / warmup_epochs

    if epoch >= lr_decay_epochs:
        return min_lr

    decay_ratio = (epoch - warmup_epochs) / (lr_decay_epochs - warmup_epochs)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, dir = out_dir)




train_dataset = AdditionDataset(train_data, block_size)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # This replaces the random sampling in get_batch
    pin_memory=(device_type=='cuda')
)

val_dataset = AdditionDataset(val_data, block_size)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,  # This replaces the random sampling in get_batch
    pin_memory=(device_type=='cuda')
)

encode, decode = get_encode_decode(meta_path, tokenizer=tokenizer)

result_dict = {'epoch': [], 'iter' : [], 'train_loss': [], 'val_loss': [], 'test_acc': [], 'train_acc': []}
if multi_digit:
    digit_accuracy_dictionary = {}
    for digit in range(1, num_digit+1):
        digit_accuracy_dictionary[f"digit_{digit}"] = []

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

eval_epoch = config.get('eval_epoch', 4)
max_iters = config.get('max_iters', 10000)
max_epochs = config.get('max_epochs', 14)
lr_decay_epochs = config.get('lr_decay_epochs', max_epochs)  # number of epochs to decay learning rate
warmup_epochs = config.get('warmup_epochs', 2)  # number of epochs to warm up learning rate



for epoch in range(max_epochs):
  model.train()

  if decay_lr:
    lr = get_lr_for_epoch(epoch)
  else:
    lr = learning_rate
  epoch_loss = 0
  samples_processed = 0
  if epoch % eval_epoch == 0:
    losses = estimate_loss()
    if eval_addition:
      config['start'] = start
      test_accuracy, *_ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=num_digit, zero_pad=zero_pad,
                                                        reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason,
                                                        binary=binary, data_type = data_type, operator=operator, data_format=data_format)
    if eval_addition_train:
        config['start'] = start_train
        train_accuracy, *_ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=num_digit, zero_pad=zero_pad,
                                                        reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason,
                                                        binary=binary, data_type=data_type, operator=operator, data_format=data_format)
    if wandb_log:
      wandb_dict = {
          "epoch": epoch,
          "train/loss": losses['train'],
          "val/loss": losses['val'],
          "lr": lr,
          "mfu": running_mfu*100, # convert to percentage,
          "test/accuracy": test_accuracy if eval_addition else None,
          "train/accuracy": train_accuracy if eval_addition_train else None
      }
      wandb.log(wandb_dict)
    result_dict['epoch'].append(epoch)
    result_dict['iter'].append(iter_num)
    result_dict['train_loss'].append(losses['train'].item())
    result_dict['val_loss'].append(losses['val'].item())
    result_dict['test_acc'].append(test_accuracy if eval_addition else None)
    result_dict['train_acc'].append(train_accuracy if eval_addition_train else None)
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
    checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'best_perplexity': best_perplexity,
                    'best_accuracy': best_accuracy,
                    'config': config,
                }
    ckpt_path_name = f'ckpt_epoch_{epoch}.pt'
    print(f"saving checkpoint to {out_dir}/{ckpt_path_name}")
    torch.save(checkpoint, os.path.join(out_dir, ckpt_path_name))

    if eval_addition and test_accuracy > best_accuracy:
          best_accuracy = test_accuracy
          checkpoint['best_accuracy'] = best_accuracy
          if iter_num > 0:
              print(f"saving checkpoint to {out_dir}/{ckpt_path_name}")
              torch.save(checkpoint, os.path.join(out_dir, ckpt_path_name.split('.pt')[0]+'_acc.pt'))
  for batch_idx, (X, Y) in enumerate(train_loader):
    X, Y = X.to(device), Y.to(device)
    with ctx:
      logits, loss = model(X, Y)
    scaler.scale(loss).backward()
    if grad_clip != 0.0:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    epoch_loss += loss.item()
    iter_num += 1
    if iter_num > max_iters:
      break
  epoch_loss /= len(train_loader)

  print(f"epoch {epoch+1}: train loss {epoch_loss:.4f}")
  if iter_num > max_iters:
    print(f"stopping after {iter_num} iterations")
    break


epoch_loss = 0
samples_processed = 0

losses = estimate_loss()
if eval_addition:
  config['start'] = start
  test_accuracy, _ , correct, incorrect = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=num_digit, zero_pad=zero_pad,
                                                    reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason,
                                                    binary=binary, data_type = data_type, operator=operator, data_format=data_format,analyze=True)
  import csv
  correct_path = os.path.join(result_dir, 'correct_examples.csv')
  with open(correct_path, 'w', newline='') as csvfile:
    fieldnames = ['a', 'b', 'c', 'chat', 'output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, nums in enumerate(correct):
        a, b, c, output, chat = nums
        writer.writerow({'a': a, 'b': b, 'c': c, 'chat': chat, 'output': output})
if eval_addition_train:
    config['start'] = start_train
    train_accuracy, *_ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=num_digit, zero_pad=zero_pad,
                                                    reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason,
                                                    binary=binary, data_type=data_type, operator=operator, data_format=data_format)
if wandb_log:
  wandb_dict = {
      "epoch": max_epochs,
      "train/loss": losses['train'],
      "val/loss": losses['val'],
      "lr": lr,
      "mfu": running_mfu*100, # convert to percentage,
      "test/accuracy": test_accuracy if eval_addition else None,
      "train/accuracy": train_accuracy if eval_addition_train else None
  }
  wandb.log(wandb_dict)
result_dict['epoch'].append(max_epochs)
result_dict['iter'].append(iter_num)
result_dict['train_loss'].append(losses['train'].item())
result_dict['val_loss'].append(losses['val'].item())
result_dict['test_acc'].append(test_accuracy if eval_addition else None)
result_dict['train_acc'].append(train_accuracy if eval_addition_train else None)
result_df = pd.DataFrame(result_dict)
result_df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'best_perplexity': best_perplexity,
                'best_accuracy': best_accuracy,
                'config': config,
            }
ckpt_path_name = f'ckpt_epoch_{max_epochs}.pt'
print(f"saving checkpoint to {out_dir}/{ckpt_path_name}")
torch.save(checkpoint, os.path.join(out_dir, ckpt_path_name))