import random
import os
import numpy as np
import pickle

def make_addition_examples_all3digits(num_samples=5000, pad=False, input_file_path='/content/drive/MyDrive/addition/add_examples.txt'):
    print('making examples of a+b=c')

    # Stratified sampling approach
    samples = 0
    with open(input_file_path, 'w+') as f:
        # Define strata (dividing the range into sections)
        strata = [
            (100, 399), # Lower range
            (400, 699), # Middle range
            (700, 999)  # Upper range (including numbers close to 999)
        ]

        # Allocate equal number of samples to each stratum combination
        samples_per_combo = num_samples // 9  # 9 combinations of strata

        for a_stratum in strata:
            for b_stratum in strata:
                for _ in range(samples_per_combo):
                    a = random.randint(a_stratum[0], a_stratum[1])
                    b = random.randint(b_stratum[0], b_stratum[1])
                    c = a + b

                    if pad:
                        f.write(f"{a:03}+{b:03}={c:04}\n")
                    else:
                        f.write(f"{a}+{b}={c}\n")

                    samples += 1

        # Add remaining samples randomly to reach exact num_samples
        remaining = num_samples - samples
        for _ in range(remaining):
            a = random.randint(100, 999)
            b = random.randint(100, 999)
            c = a + b

            if pad:
                f.write(f"{a:03}+{b:03}={c:04}\n")
            else:
                f.write(f"{a}+{b}={c}\n")

            samples += 1

    print(f'{samples} number of samples saved to {input_file_path}')


def generate_partial_carry_pairs(input_file_path=None,num_pairs=10000):
    """
    Generate pairs of three-digit integers where addition only allows carry in the hundreds place.
    This means that the sum of ones and tens digits must be <= 9, but hundreds can exceed 9.
    """

    num = 0
    with open(input_file_path, 'w+') as f:
      while num < num_pairs:
          # Generate first number (a)
          a_hundreds = random.randint(1, 9)
          a_tens = random.randint(0, 9)
          a_ones = random.randint(0, 9)

          # Calculate maximum allowed digits for b to avoid carry in ones and tens
          b_max_tens = 9 - a_tens
          b_max_ones = 9 - a_ones

          # For hundreds place, we can use any digit from 1-9
          # Generate second number (b)
          b_hundreds = random.randint(1, 9)
          b_tens = random.randint(0, b_max_tens)
          b_ones = random.randint(0, b_max_ones)

          # Construct the full numbers
          a = a_hundreds * 100 + a_tens * 10 + a_ones
          b = b_hundreds * 100 + b_tens * 10 + b_ones

          # Add the pair to our list

          f.write(f"{a}+{b}=\n")
          num+=1

def generate_addition_with_tens_hundreds_carry(input_file_path='three_digit_addition.txt', num_pairs=10000):
    """
    Generate pairs of three-digit integers where addition allows carry in both tens and hundreds places.
    Only the ones digit will not carry.
    """
    num = 0
    with open(input_file_path, 'w+') as f:
        while num < num_pairs:
            # Generate first number (a)
            a_hundreds = random.randint(1, 9)
            a_tens = random.randint(0, 9)
            a_ones = random.randint(0, 9)

            # Calculate maximum allowed digits for b to avoid carry in ones only
            b_max_ones = 9 - a_ones

            # For tens and hundreds place, we can use any valid digit
            # Generate second number (b)
            b_hundreds = random.randint(1, 9)
            b_tens = random.randint(0, 9) # Allow any tens digit
            b_ones = random.randint(0, b_max_ones)

            # Construct the full numbers
            a = a_hundreds * 100 + a_tens * 10 + a_ones
            b = b_hundreds * 100 + b_tens * 10 + b_ones

            # Calculate sum
            c = a + b

            # Add the pair to our output file
            f.write(f"{a}+{b}=\n")
            num += 1

    print(f"Generated {num_pairs} addition examples with carries in tens and hundreds places")



def reverse(s):
  return s[::-1]

def generate_reverse_examples(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        data = f.read()
    lines = []
    for line in data.split("\n"):
        if(line == ""):
            continue
        a = line.split("+")[0]
        b = line.split("+")[1].split("=")[0]
        c = line.split("=")[1]
        c = reverse(c)
        a, b = int(a), int(b)
        lines.append(f"{a}+{b}={c}\n")

    with open(output_file_path, "w+") as f2:
        for line in lines:
            f2.write(line)


def create_meta_data(data, output_file_path):
  chars = sorted(list(set(data)))
  vocab_size = len(chars)
  print("all the unique characters:", ''.join(chars))
  print(f"vocab size: {vocab_size:,}")

  # create a mapping from characters to integers
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }

  def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
  def decode(l):
    ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

  n = len(data)
  print("Number of character:" + str(n))
  train_data = data[:int(n*0.9)]
  val_data = data[int(n*0.9):]

  train_ids = encode(train_data)
  val_ids = encode(val_data)
  train_ids = np.array(train_ids, dtype=np.uint16)
  val_ids = np.array(val_ids, dtype=np.uint16)
  os.makedirs(output_file_path, exist_ok=True)
  train_ids.tofile(f'{output_file_path}/train.bin')
  val_ids.tofile(f'{output_file_path}/val.bin')


# save the meta information as well, to help us encode/decode later
  meta = {
      'vocab_size': vocab_size,
      'itos': itos,
      'stoi': stoi,
  }

  with open(f'{output_file_path}/meta.pkl', 'wb') as f:
      pickle.dump(meta, f)

