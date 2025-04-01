from main_utilities import *
import tqdm
import torch
import numpy as np
import random
import math
import os
import pandas as pd


def get_abc_new(abc: str, zero_pad=False, reverse_ab=False, binary=False, few_shot=False, algo_reason=False):
    if 'sin(' in abc:
        operation = 'sin'
    elif 'sqrt(' in abc:
        operation = 'sqrt'
    elif '+' in abc:
        operation = '+'
    elif '-' in abc:
        operation = '-'
    elif '*' in abc:
        operation = '*'

    else:
        print(f'operation not found, abc: {abc}')

    if operation in ['+', '-', '*']:
        [a,b] = abc.split(operation)

    elif operation in ['sin', 'sqrt']:
        if 'Input:' in abc:
            a = abc.split('Input:\n')[-1].split('\nTarget')[0]
        else:
            # a, _ = abc.strip().split('=')
            a = abc.strip().split('=')[0]
        a = a.replace(operation, '').replace('(', '').replace(')', '')
        b = ''

    if a[0] == '$':
        a = a[1:]
    if a.startswith('Input:\n'):
        a = a.split('Input:\n')[-1]
    if 'Target' in b:
        b = b.split('\nTarget')[0]

    b = b.split('=')[0]

    a = a.replace(' ', '')
    b = b.replace(' ', '')

    if binary:
        c = int(a,2) + int(b,2)
        return a, b, int(convert_to_binary(c))

    if zero_pad:
        a, b = remove_zero_pad(a), remove_zero_pad(b)

    if reverse_ab:
        a, b = reverse_string(a), reverse_string(b)

    if operation == '+': c = int(a) + int(b)

    if '\n' in b: b = b[:-1]

    return a,b,c,operation

def get_data_list(filename=None, operator='+', delim=None):
    import re
    data_list = []
    if filename: # read data from file
        if operator in ['text']:
            with open(filename, 'r') as f:
                data = f.read()
            data_splitted = data.split('\n\n')
            for line in data_splitted:
                data_list.append((line, operator))
        else:
            with open(filename, 'r') as f:
                lines = f.readlines()
            for line in lines:
                # if first char is $, assume it's a delimiter
                if line[0] == '$':
                    delim = '$'
                if delim:
                    # remove delim from line
                    line = line.replace(delim, '')
                # x1, x2 = line.strip().split(operator)
                if operator in ['+', '-', '*']:
                    x1, x2 = re.split(r'[+\-\*]', line.strip())
                    x2, y = x2.split("=")
                    if operator == '+':
                        y2 = int(x1) + int(x2)
                    elif operator == '-':
                        y2 = int(x1) - int(x2)
                    elif operator == '*':
                        y2 = int(x1) * int(x2)

                    data_list.append((int(x1), int(x2), int(y2), operator))

                elif operator in ['sin', 'sqrt']:
                    x = line.strip().split('=')[0]
                    x = x.replace(operator, '').replace('(', '').replace(')', '')
                    # x = re.findall(r'\d+', x)
                    # x = '.'.join(x)
                    # y = line.strip().split('=')[1]
                    if operator == 'sin':
                        y = math.sin(float(x))
                    elif operator == 'sqrt':
                        y = math.sqrt(float(x))
                    y = math.floor(y * 10000) / 10000

                    data_list.append((float(x), float(y), operator))


    else: # generate random data
        if operator in ['text']:
            # TODO: For now for creating validation dataset, we just use the last 10% of the shakespeare dataset
            with open('data/shakespeare/input.txt', 'r') as f:
                data = f.read()
                n_text = len(data)
                data = data[int(n_text*0.9):]
            data_splitted = data.split('\n\n')
            for line in data_splitted:
                data_list.append((line, operator))
        else:
            for _ in range(1000):
                if operator in ['+', '-', '*']:
                    x1, x2 = random.randint(0, 999), random.randint(0, 999)
                    if operator == '+':
                        y = x1 + x2
                    elif operator == '-':
                        y = x1 - x2
                    elif operator == '*':
                        y = x1 * x2
                    data_list.append((int(x1), int(x2), int(y), operator))

                elif operator in ['sin', 'sqrt']:
                    if operator == 'sin':
                        x = random.uniform(-math.pi/2, math.pi/2)
                        x = math.floor(x * 10000) / 10000
                        y = math.sin(x)
                    elif operator == 'sqrt':
                        x = random.uniform(0, 10)
                        x = math.floor(x * 10000) / 10000
                        y = math.sqrt(x)

                    y = math.floor(y * 10000) / 10000

                    data_list.append((float(x), float(y), operator))

    return data_list

_precomputed_batches = {}
def prepare_addition_batches(config, encode, num_digit=3, zero_pad=False, reverse_ab=False, binary=False, fewshot=False,
                           algo_reason=False, data_type='binary', operator='+', data_format='plain', add_space=False, simple=False, random_A=False, random_C=False):
    device = config['device']
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    print(f"Preparing batches from: {start}")
    if data_type=='text':
        test_data_file = start[5:]
        print(f"Preparing batches using test data file: {test_data_file}")
        test_data_list = get_data_list(test_data_file, operator=operator)
        test_data_str = generate_data_str(test_data_list, operator=operator, format=data_format, train=False, shuffle=True,
                                          add_space=add_space, simple=simple, random_A=random_A, random_C=random_C)
        if algo_reason:
            lines = [x.strip() + "\nTarget:\n" for x in test_data_str.split("Target:")]
            lines = lines[:-1]
        else:
            lines = test_data_str.split('\n')[:-1]
    else:
        # encode the beginning of the prompt
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                lines = [line.rstrip() for line in f]
                if algo_reason:
                    lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
                    lines = lines2
        else:
            lines = start.splitlines()
            if algo_reason:
                lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
                lines = lines2

    total = len(lines)

    print(f'Preparing batches for {total} examples from: {start}')
    #Process all lines and group by prompt length
    prompt_dict = {}
    for line_idx in range(total):
        line = lines[line_idx]
        line = line.strip('\n')
        start_ids = encode(line)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        a, b, c, op = get_abc_new(line, zero_pad=zero_pad, reverse_ab=reverse_ab, binary=binary, few_shot=fewshot, algo_reason=algo_reason)
        a_d, b_d, num_carry = get_num_digits(a), get_num_digits(b), numCarryOps(a, b, binary=binary)
        prompt_length = len(start_ids)
        input_tuple = (x, len(line), line[0], a, b, c, a_d, b_d, num_carry)

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
    config_hash = hash(frozenset({k: str(v) for k, v in config.items() if k != 'device'}.items()))
    batch_key = f"{config_hash}_{data_type}_{operator}_{num_digit}_{zero_pad}_{reverse_ab}_{binary}_{fewshot}_{algo_reason}_{data_format}_{add_space}_{simple}_{random_A}_{random_C}"
    _precomputed_batches[batch_key] = (batch_list, total)
    return batch_list, total

# Modified evaluation function that uses pre-created batches
# Modified evaluation function that uses pre-created batches
def evaluate_addition_precomputed(config, model, ctx, decode, batch_list, total,
                                  verbose=False, num_digit=3, zero_pad=False, reverse_c=False,
                                  algo_reason=False, binary=False, add_space=False, simple=False, operator='+', verbose_correct=False, analyze=False):
    model.eval()
    device = config['device']
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+2
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200
    eps = config['eps'] if 'eps' in config.keys() else 0

    if algo_reason:
        max_new_tokens = 80 if (simple or ('simple' in config.get('dataset', ''))) else 320
    elif add_space:
        max_new_tokens = 2 * num_digit + 3

    if 'multi_digit' in config.keys() and config['multi_digit']:
        if algo_reason and not simple:
            if num_digit == 1: max_new_tokens = 160
            elif num_digit == 2: max_new_tokens = 220
            elif num_digit == 3: max_new_tokens = 290
            elif num_digit == 4: max_new_tokens = 370
            elif num_digit == 5: max_new_tokens = 450
            elif num_digit == 6: max_new_tokens = 540
            elif num_digit == 7: max_new_tokens = 630
            elif num_digit == 8: max_new_tokens = 800
            else: max_new_tokens = 1000
        if algo_reason and simple:
            max_new_tokens = 20 + 15 * num_digit

    correct = 0
    carry_dictionary = {f'carry{i}_correct': 0 for i in range(num_digit+1)}
    carry_dictionary.update({f'carry{i}_total': 0 for i in range(num_digit+1)})

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
                    _, len_x, line_start, a, b, c, a_d, b_d, num_carry = batch[i]
                    c_hat = outcome[len_x:]

                    # Process the output to get the actual prediction
                    if '$' == line_start:  # handle $ prompt $
                        c_hat = c_hat.split('$')[0]
                    else:
                        if '\n' == c_hat[-1]:  # handle cases where it ends with '\n'
                            c_hat = c_hat[:-1]

                    c_hat2 = c_hat
                    if zero_pad:
                        c_hat2 = remove_zero_pad(c_hat)

                    if algo_reason:
                        if '</scratch>\n' in c_hat:
                            c_hat2 = c_hat.split('</scratch>\n')[1].split('\n')[0]
                            c_hat2 = c_hat2.replace(' ', '')
                        if ('simple' in config.get('dataset', '') or config.get('simple', False)) and '.\n' in c_hat:
                            c_hat2 = c_hat2.split('.\n')[1]
                            c_hat2 = c_hat2.split('\n')[0]
                    else:  # plain addition
                        c_hat2 = c_hat2.split('\n')[0]

                    if reverse_c:
                        c_hat2 = reverse_string(c_hat2)

                    if add_space:
                        c_hat2 = c_hat2.replace(' ', '')

                    if is_number(c_hat2):
                        if '.' in c_hat2:
                            c_hat2 = float(c_hat2)
                        else:
                            c_hat2 = int(c_hat2)
                    else:  # c_hat2 is not a number
                        c = str(c)

                    # Check correctness
                    if op in ['+', '-', '*']:
                        if c == c_hat2:
                            correct += 1
                            carry_dictionary[f'carry{num_carry}_correct'] += 1
                            correct_examples.append((a, b, c, outcome, c_hat2))
                            if verbose_correct:
                                print('outputs(o): ', outcome)
                                print(f'correct: {a}{op}{b}={c}')
                        else:
                            incorrect_examples.append((a, b, c, outcome, c, c_hat2))
                            if verbose:
                                print('outputs(x): ', outcome)
                                print(f'wrong  : {a}{op}{b}={c_hat2}')
                                print(f'correct: {a}{op}{b}={c}')
                    elif op in ['sin', 'sqrt']:
                        if type(c) != str and abs(c - c_hat2) <= eps:
                            correct += 1
                            carry_dictionary[f'carry{num_carry}_correct'] += 1
                            if verbose_correct:
                                print('outputs(o): ', outcome)
                                print(f'correct: {op}({a})={c}')
                        else:
                            if verbose:
                                print('outputs(x): ', outcome)
                                print(f'wrong  : {op}({a})={c_hat2}')
                                print(f'correct: {op}({a})={c}')

                    carry_dictionary[f'carry{num_carry}_total'] += 1

                    # Calculate metrics if analyzing
                    if analyze:
                        error_dict['y'].append(c)
                        error_dict['y_hat'].append(c_hat2)

                        metric_types = ['mse', 'normalized_mse', 'digit_wise_difference', 'incorrect_digit_count']
                        for metric_type in metric_types:
                            error, list_not_num, list_outlier_num = get_error_metric(c, c_hat2, metric_type, eps=config.get('eps', 0),
                                                                                    list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                            error_dict[f'{metric_type}'].append(error)

                        error, _, _ = get_error_metric(c, c_hat2, 'accuracy', eps=0, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps0'].append(error * 100)
                        error, _, _ = get_error_metric(c, c_hat2, 'accuracy', eps=5e-4, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps5e-4'].append(error * 100)
                        error, _, _ = get_error_metric(c, c_hat2, 'accuracy', eps=5e-3, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps5e-3'].append(error * 100)

    accuracy = correct / total * 100
    print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")

    accuracy_dictionary = {
        f'carry{i}': carry_dictionary[f'carry{i}_correct'] / carry_dictionary[f'carry{i}_total'] * 100
        if carry_dictionary[f'carry{i}_total'] != 0 else np.nan
        for i in range(num_digit+1)
    }
    print(accuracy_dictionary)

    if analyze:
        error_df = pd.DataFrame(error_dict)
        result_dir = config.get('result_dir', get_results_dir(config))
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
        print('skipped since not a number: ', list_not_num)
        print('skipped since outlier number: ', list_outlier_num)

    model.train()
    return accuracy, accuracy_dictionary, correct_examples, incorrect_examples

# Keep the original function for backward compatibility, but make it use the new functions
def evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=3, zero_pad=False, reverse_ab=False, reverse_c=False,
                           algo_reason=False, binary=False, fewshot=False, data_type='binary', operator='+', data_format='plain', verbose_correct=False, analyze=False):
    config_hash = hash(frozenset({k: str(v) for k, v in config.items() if k != 'device'}.items()))
    batch_key = f"{config_hash}_{data_type}_{operator}_{num_digit}_{zero_pad}_{reverse_ab}_{binary}_{fewshot}_{algo_reason}_{data_format}_{config.get('add_space', False)}_{config.get('simple', False)}_{config.get('random_A', False)}_{config.get('random_C', False)}"
    if batch_key in _precomputed_batches:
        print("Using precomputed batches")
        batch_list, total = _precomputed_batches[batch_key]
    else:
      print("Creating new batches")
      batch_list, total = prepare_addition_batches(
        config, encode, num_digit=num_digit, zero_pad=zero_pad, reverse_ab=reverse_ab,
        binary=binary, fewshot=fewshot, algo_reason=algo_reason, data_type=data_type,
        operator=operator, data_format=data_format, add_space=config.get('add_space', False),
        simple=config.get('simple', False), random_A=config.get('random_A', False),
        random_C=config.get('random_C', False)
    )

    # Evaluate using the batches
    return evaluate_addition_precomputed(
        config, model, ctx, decode, batch_list, total, verbose=verbose,
        num_digit=num_digit, zero_pad=zero_pad, reverse_c=reverse_c,
        algo_reason=algo_reason, binary=binary, add_space=config.get('add_space', False),
        simple=config.get('simple', False), operator=operator,
        verbose_correct=verbose_correct, analyze=analyze
    )