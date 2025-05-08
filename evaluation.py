from main_utilities import *
from tqdm.auto import tqdm
import torch
import numpy as np
import random
import math
import os
import pandas as pd


def get_abc_new(abc: str, zero_pad=False, reverse_ab=False, binary=False):
    """Parse addition expression and return operands and result.
    Args:
        abc: String in format "a+b+c...=result"
        zero_pad: Whether to remove zero padding
        reverse_ab: Whether to reverse operands
        binary: Whether operands are in binary
    Returns:
        tuple: (operands_str, result, operation)
    """
    if '+' in abc:
        operation = '+'
    elif '-' in abc:
        operation = '-' 
    elif '*' in abc:
        operation = '*'
    else:
        print(f'operation not found, abc: {abc}')
        return None, None, None

    # Split the input string into parts
    parts = abc.split('=')
    if len(parts) != 2:
        print(f'Invalid format, expected "a+b+c...=result", got: {abc}')
        return None, None, None

    # Get the operands part (before =)
    operands_str = parts[0]
    if operands_str[0] == '$':
        operands_str = operands_str[1:]
    if operands_str.startswith('Input:\n'):
        operands_str = operands_str.split('Input:\n')[-1]
    if 'Target' in operands_str:
        operands_str = operands_str.split('\nTarget')[0]

    # Split into individual operands
    operands = [op.strip() for op in operands_str.split('+')]
    
    # Clean up operands
    operands = [op.replace(' ', '') for op in operands]
    
    if binary:
        # Convert all operands to binary and sum
        result = sum(int(op, 2) for op in operands)
        return operands_str, result, operation

    if zero_pad:
        operands = [remove_zero_pad(op) for op in operands]

    if reverse_ab:
        operands = [reverse_string(op) for op in operands]

    if operation == '+':
        result = sum(int(op) for op in operands)

    return operands_str, result, operation

_precomputed_batches = {}
def prepare_addition_batches(config, encode, num_digit=3, zero_pad=False, reverse_ab=False, binary=False,  data_type='binary', operator='+', data_format='plain', add_space=False, simple=False):
    device = config['device']
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    print(f"Preparing batches from: {start}")
    
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]
    else:
        lines = start.splitlines()

    total = len(lines)
    print(f'Preparing batches for {total} examples from: {start}')
    
    # Process all lines and group by prompt length
    prompt_dict = {}
    for line_idx in range(total):
        line = lines[line_idx]
        line = line.strip('\n')
        start_ids = encode(line)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        operands, result, op = get_abc_new(line, zero_pad=zero_pad, reverse_ab=reverse_ab, binary=binary)
        prompt_length = len(start_ids)
        input_tuple = (x, len(line), line[0], operands, result)

        if prompt_length in prompt_dict.keys():
            prompt_dict[prompt_length].append(input_tuple)
        else:
            prompt_dict[prompt_length] = [input_tuple]

    # Construct batches of prompts
    batch_list = []
    for prompt_length in prompt_dict.keys():
        input_tuple_list = prompt_dict[prompt_length]
        for batch_idx in range(math.ceil(len(input_tuple_list)/test_batch_size)):
            batch_list.append(input_tuple_list[batch_idx*test_batch_size:(batch_idx+1)*test_batch_size])

    print(f'Created {len(batch_list)} batches')
    
    # Cache the batches using a hash of the configuration
    config_hash = hash(frozenset({k: str(v) for k, v in config.items() if k != 'device'}.items()))
    batch_key = f"{config_hash}_{data_type}_{operator}_{num_digit}_{zero_pad}_{reverse_ab}_{binary}__{data_format}_{add_space}_{simple}"
    _precomputed_batches[batch_key] = (batch_list, total)
    
    return batch_list, total

# Modified evaluation function that uses pre-created batches
def evaluate_addition_precomputed(config, model, ctx, decode, batch_list, total,
                                  verbose=False, num_digit=3, zero_pad=False, reverse_c=False,
                                  add_space=False, operator='+', verbose_correct=False, analyze=False):
    model.eval()
    device = config['device']
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+2
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200

    if add_space:
        max_new_tokens = 2 * num_digit + 3

    correct = 0

    if analyze:
        # analyze various metrics
        error_dict = {'y': [], 'y_hat': [], 'accuracy_eps0': [], 'accuracy_eps5e-4': [],
                      'accuracy_eps5e-3': [], 'mse': [], 'normalized_mse': [],
                      'digit_wise_difference': [], 'incorrect_digit_count': []}
        list_not_num = []
        list_outlier_num = []
    op = operator
    correct_examples = []
    incorrect_examples = []
    print(f"Max number of tokens {max_new_tokens}.")
    for batch_idx in tqdm(range(len(batch_list))):
        batch = batch_list[batch_idx]
        x_list = [input_tuple[0] for input_tuple in batch]
        x = torch.cat(x_list, dim=0)

        # Run generation
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                outcome_list = [decode(y_i.tolist()) for y_i in y]

                for i, outcome in enumerate(outcome_list):
                    _, len_x, line_start, operands, result = batch[i]
                    print(f"\nDebug - Full outcome: {outcome}")
                    print(f"Debug - Input length (len_x): {len_x}")
                    print(f"Debug - Raw generated tokens: {outcome[len_x:]}")
                    
                    # The model will never generate more than max_new_tokens
                    c_hat = outcome[len_x:]
                    print(f"Debug - After len_x slice: {c_hat}")

                    # Process the output to get the actual prediction
                    if '$' == line_start:  # handle $ prompt $
                        c_hat = c_hat.split('$')[0]
                        print(f"Debug - After $ split: {c_hat}")
                    else:
                        if '\n' == c_hat[-1]:  # handle cases where it ends with '\n'
                            c_hat = c_hat[:-1]
                            print(f"Debug - After newline removal: {c_hat}")

                    c_hat2 = c_hat
                    if zero_pad:
                        c_hat2 = remove_zero_pad(c_hat)
                        print(f"Debug - After zero pad removal: {c_hat2}")

                    # plain addition
                    c_hat2 = c_hat2.split('\n')[0]
                    print(f"Debug - After newline split: {c_hat2}")

                    if reverse_c:
                        c_hat2 = reverse_string(c_hat2)
                        print(f"Debug - After reverse: {c_hat2}")

                    if add_space:
                        c_hat2 = c_hat2.replace(' ', '')
                        print(f"Debug - After space removal: {c_hat2}")

                    if is_number(c_hat2):
                        if '.' in c_hat2:
                            c_hat2 = float(c_hat2)
                        else:
                            c_hat2 = int(c_hat2)
                    else:  # c_hat2 is not a number
                        result = str(result)

                    # Check correctness
                    if op in ['+', '-', '*']:
                        if result == c_hat2:
                            correct += 1
                            correct_examples.append((operands, result, outcome, c_hat2))
                            if verbose_correct:
                                print('outputs(o): ', outcome)
                                print(f'correct: {operands}={result}')
                        else:
                            incorrect_examples.append((operands, result, outcome, c_hat2))
                            if verbose:
                                print('outputs(x): ', outcome)
                                print(f'wrong  : {operands}={c_hat2}')
                                print(f'correct: {operands}={result}')
                    # Calculate metrics if analyzing
                    if analyze:
                        error_dict['y'].append(result)
                        error_dict['y_hat'].append(c_hat2)

                        metric_types = ['mse', 'normalized_mse', 'digit_wise_difference', 'incorrect_digit_count']
                        for metric_type in metric_types:
                            error, list_not_num, list_outlier_num = get_error_metric(result, c_hat2, metric_type, eps=config.get('eps', 0),
                                                                                    list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                            error_dict[f'{metric_type}'].append(error)

                        error, _, _ = get_error_metric(result, c_hat2, 'accuracy', eps=0, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps0'].append(error * 100)
                        error, _, _ = get_error_metric(result, c_hat2, 'accuracy', eps=5e-4, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps5e-4'].append(error * 100)
                        error, _, _ = get_error_metric(result, c_hat2, 'accuracy', eps=5e-3, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps5e-3'].append(error * 100)

    accuracy = correct / total * 100
    print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")

    accuracy_dictionary = {}
    if analyze:
        error_df = pd.DataFrame(error_dict)
        result_dir = config.get('result_dir')
        if result_dir is None:
            result_dir = get_results_dir(config)
        error_df.to_csv(os.path.join(result_dir, 'error_df.csv'), index=False)

        error_mean_dict = {
            metric_type: np.nanmean(error_dict[f'{metric_type}'])
            for metric_type in ['accuracy_eps0', 'accuracy_eps5e-4', 'accuracy_eps5e-3',
                               'mse', 'normalized_mse', 'digit_wise_difference', 'incorrect_digit_count']
        }
        error_mean_dict['num_not_num'] = len(list_not_num) / len(metric_types)
        error_mean_dict['num_outlier_num'] = len(list_outlier_num) / len(metric_types)
        error_mean_dict['median_mse'] = error_df.mse.median()
        error_mean_dict['median_normalized_mse'] = error_df.normalized_mse.median()
        accuracy_dictionary.update(error_mean_dict)

    model.train()
    return accuracy, accuracy_dictionary, correct_examples, incorrect_examples

# Keep the original function for backward compatibility, but make it use the new functions
def evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=3, zero_pad=False, 
                          reverse_ab=False, reverse_c=False, data_type='binary', operator='+', 
                          data_format='plain', add_space=False, verbose_correct=False, analyze=False):
    config_hash = hash(frozenset({k: str(v) for k, v in config.items() if k != 'device'}.items()))
    batch_key = f"{config_hash}_{data_type}_{operator}_{num_digit}_{zero_pad}_{reverse_ab}_False_{data_format}_{add_space}_False"
    
    if batch_key in _precomputed_batches:
        print("Using precomputed batches")
        batch_list, total = _precomputed_batches[batch_key]
    else:
        print("Creating new batches")
        batch_list, total = prepare_addition_batches(
            config, encode, num_digit=num_digit, zero_pad=zero_pad, reverse_ab=reverse_ab,
            data_type=data_type, operator=operator, data_format=data_format, add_space=add_space
        )

    # Evaluate using the batches
    return evaluate_addition_precomputed(
        config, model, ctx, decode, batch_list, total, verbose=verbose,
        num_digit=num_digit, zero_pad=zero_pad, reverse_c=reverse_c,
        add_space=add_space, operator=operator, verbose_correct=verbose_correct, analyze=analyze
    )