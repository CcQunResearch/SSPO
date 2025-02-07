from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import re
import string
import random
import numpy as np
from llm_generate import textGenerator
from post_tts import tts
from util import *
import argparse
import yaml
import os.path as osp

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--k_num', type=int, required=True, default=1, help='index')
parser.add_argument('--cuda', type=int, required=True, default=0, help='cuda')
parser.add_argument('--all_k', type=int, required=True, default=1, help='iter num')

args = parser.parse_args()
k_num = args.k_num + 1
all_k = args.all_k
temp_save_path = "temp/"

dirname = osp.dirname(osp.abspath(__file__))
config_path = osp.join(dirname, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
# save_path = config['dpo_sampling']['save_path']
model_id = config['dpo_sampling']['model_name_or_path']
dpo_dataset_path = config['dpo_sampling']['dpo_dataset_path']
template = config['dpo_sampling']['template']
max_token_nums = config['dpo_sampling']['max_token_nums']
reversa_flag = config['dpo_sampling']['reversa_flag']

gpu = str(args.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

text_generator = textGenerator(model_id)


def penalty(x_duration, y_duration, alpha=1, beta=0.8):
    term1 = alpha * (np.exp(max(0, y_duration - x_duration)) - 1)
    term2 = beta * max(0, x_duration - y_duration)
    return term1 + term2


def select_en(select_zh, ens):
    start = time.time()
    chose_en = None
    reject_en = None
    min_penalty = float('inf')
    max_penalty = float('-inf')
    closest_en = None
    farthest_en = None
    min_dur = float('inf')
    max_dur = float('-inf')

    en_durations = [None] * len(ens)
    futures = []
    text_list = ens + [select_zh]
    with ThreadPoolExecutor(max_workers=5) as executor:
        for idx, text in enumerate(text_list):
            future = executor.submit(tts, text)
            futures.append((idx, future))

    for idx, future in futures[:-1]:
        en_durations[idx] = future.result()
    zh_dur = futures[-1][1].result()
    for index, (en, en_duration) in enumerate(zip(ens, en_durations)):
        en_penalty = penalty(zh_dur, en_duration)
        if reversa_flag:
            if chose_en is None or en_penalty >= max_penalty:
                chose_en = en
                chose_index = index
                max_penalty = en_penalty
            if reject_en is None or en_penalty < min_penalty:
                reject_en = en
                reject_index = index
                min_penalty = en_penalty
        else:
            if chose_en is None or en_penalty <= min_penalty:
                chose_en = en
                chose_index = index
                min_penalty = en_penalty
            if reject_en is None or en_penalty > max_penalty:
                reject_en = en
                reject_index = index
                max_penalty = en_penalty

        if closest_en is None or en_duration < min_dur:
            closest_en = en
            close_index = index
            min_dur = en_duration
        if farthest_en is None or en_duration > max_dur:
            farthest_en = en
            far_index = index
            max_dur = en_duration

    return chose_en, reject_en, chose_index, reject_index, zh_dur, en_durations, closest_en, farthest_en, close_index, far_index


def robust_select_en(select_zh, new_ens):
    i = 0
    while True:
        i += 1
        if i > 10:
            return None, None, None, None, None, None, None, None, None, None
        try:
            # time
            chose_en, reject_en, chose_index, reject_index, zh_dur, en_durations, closest_en, farthest_en, close_index, far_index = select_en(
                select_zh, new_ens)

            if -1 == zh_dur or -1 in en_durations:
                print(f'-----tts have -1-------')
            else:
                return chose_en, reject_en, chose_index, reject_index, zh_dur, en_durations, closest_en, farthest_en, close_index, far_index
        except Exception as e:
            print("Exception occurred: ", e)


def robust_quest(prompt, query, src_zh, min_sample_num=4):
    i = 0
    ens = []
    while True:
        i += 1
        if i > 2:
            print(f'----[robust_quest]---sample num over----###len(ens):{len(ens)}')
            if not maxlen_flag:
                return ''
            else:
                return ens
        max_token_num = max_token_nums + (i - 1) * 16
        max_topk = min(40 + i * 20, 100)
        tem = min(1.2 + i * 0.2, 2.0)
        answer = text_generator.generate(prompt, temperature=tem, top_k=max_topk, max_gen_tokens=max_token_num)

        zhs = []
        allen = []
        maxlen_flag = True
        for ans in answer:
            allen.append(ans[len(prompt):])
            maxlen_flag, en_res = find_en(ans[len(prompt):])
            if not maxlen_flag:
                print(f'zh:{src_zh},ta:{ans[len(prompt):]}')
            ############ change
            # en_res = remove_non_chinese_chars_and_brackets(en_res) #es2zh

            ens.append(en_res)
        ens = [en for en in ens if en is not None and en != ""]
        ens = list(set(ens))
        if len(ens) >= min_sample_num and maxlen_flag:
            print(f'----[robust_quest]---get enough sample---#####len(ens):{len(ens)}')
            return ens


def sample(instruct):
    if template == 'qwen':
        query = f'system\nYou are a helpful assistant.\nuser\n{instruct}\nassistant\n'
    else:
        query = instruct
    src_inputs = find_input(instruct)
    sampling_records = []
    prompt = query + src_inputs[0] + '('
    accept = ''
    reject = ''
    temp_prompt = ''
    # i = 0
    start = time.time()
    for i in range(len(src_inputs)):
        same_flag = False
        new_ens = robust_quest(prompt, instruct, src_inputs[i])
        if new_ens == '':
            return ''
        if len(new_ens) <= 3:
            same_flag = True
        if len(new_ens) == 0:
            print(f'---------error1---generate---over time--')
            return ''

        chose_en, reject_en, chose_index, reject_index, zh_dur, en_dur, closest_en, farthest_en, close_index, far_index = robust_select_en(
            src_inputs[i], new_ens)
        if chose_en == None and en_dur == None:
            print(f'---------error2--tts---over time-chose_en:{chose_en}-----en_dur:{en_dur}-')
            return ''

        if abs(min(en_dur) - max(en_dur)) < 0.08:
            same_flag = True
        sampling_records.append({
            'prompt_index': i,
            'temp_prompt': temp_prompt,
            # 'answer': answer,
            'src': src_inputs[i],
            'src_lang': 'zh',
            'tar': new_ens,
            'tar_lang': 'en',
            'chosen': chose_en,
            'chosen_index': chose_index,
            'rejected': reject_en,
            'rejected_index': reject_index,
            'src_duration': zh_dur,
            'tar_duration': en_dur,
            'same_flag': same_flag
        })

        prompt += chose_en + ')' + '\n' + src_inputs[i + 1] + '(' if i + 1 < len(src_inputs) else chose_en + ')'
        temp_prompt += src_inputs[i] + '(' + chose_en + ')' + '\n'
        accept += src_inputs[i] + '(' + chose_en + ')' + '\n'
        reject += src_inputs[i] + '(' + reject_en + ')' + '\n'
        i += 1
    print(f'------all time:{time.time() - start}')
    return {'src_prompt': instruct, 'sampling_records': sampling_records, 'num': len(sampling_records),
            'single': {'accept': accept, 'reject': reject}}


data = []
with open(dpo_dataset_path, "r") as fr:
    for temp in json.load(fr):
        data.append(temp['instruction'])

random.seed(42)
random.shuffle(data)
ck_size = len(data) // all_k
start = (k_num - 1) * ck_size
end = start + ck_size

if k_num == all_k:
    end = len(data)
for text in tqdm.tqdm(data[start:end]):
    result = sample(text)
    if result != '':
        append_to_json_and_jsonl(f'{temp_save_path}/final_res_{str(k_num)}k.json',
                                 f'{temp_save_path}/final_res_{str(k_num)}k.jsonl', result)
