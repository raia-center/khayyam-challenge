import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import itertools
import json
from prompt import *

def post_process(row):
    return ''.join(row.split('جواب:')[1:])
        
def eval_mgpt(df, prompt_type=0, load_cache=False):
    if not load_cache:
        model_name = 'ai-forever/mGPT'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name, device_map='cuda')
    
    results = []
    cache = f'cache/mgpt_{prompt_type}'
    cache_dir = os.listdir(cache)
    
    for _, i in tqdm(df.iterrows()):
        
        if f"mgpt_{prompt_type}_{i['ID']}_prompt.txt" in cache_dir and f"mgpt_{prompt_type}_{i['ID']}_str.txt" in cache_dir and f"mgpt_{prompt_type}_{i['ID']}_prob.json" in cache_dir:
            with open(os.path.join(cache, f"mgpt_{prompt_type}_{i['ID']}_prompt.txt"), 'r', encoding='utf-8') as f:
                prompt = f.read()
            with open(os.path.join(cache, f"mgpt_{prompt_type}_{i['ID']}_str.txt"), 'r', encoding='utf-8') as f:
                outputs = f.read()
            outputs = post_process(outputs)
            results.append([prompt, outputs, i['ID'], f"{cache}/mgpt_{prompt_type}_{i['ID']}_prob.json", 'mgpt'])

        elif not load_cache:
            prompt = generate_prompt(i)

            inputs = tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to('cuda')
            outputs = model.generate(inputs, temperature=0, min_length=1, max_new_tokens=1024, return_dict_in_generate=True, output_scores=True)
            
            candidate_token = ['1', '2', '3', '4', '۱', '۲', '۳', '۴', 'الف', 'ب', 'ج', 'د', 'a', 'b', 'c', 'd', str(i["Choice 1"]), str(i["Choice 2"]), str(i["Choice 3"]), str(i["Choice 4"])]
            candidate_token_id = []

            for j in candidate_token:
                candidate_token_id.append(tokenizer.encode(j))

            first_token_probs = torch.softmax(outputs.scores[0], dim=-1)
        
            prob_dict = {}
       
            for j in list(set(itertools.chain(*candidate_token_id))):
                prob_dict[tokenizer.decode(j)] = first_token_probs[0][j].item()


            out_str = tokenizer.decode(outputs.sequences[0])
            
            with open(os.path.join(cache, f"mgpt_{prompt_type}_{i['ID']}_prompt.txt"), 'w', encoding="utf-8") as f:
                f.write(prompt)

            with open(os.path.join(cache, f"mgpt_{prompt_type}_{i['ID']}_str.txt"), 'w', encoding="utf-8") as f:
                f.write(out_str)

            with open(os.path.join(cache, f"mgpt_{prompt_type}_{i['ID']}_prob.json"), 'w', encoding='utf-8') as f:
                json.dump(prob_dict, f, ensure_ascii=False, indent=4)

            out_str = post_process(out_str)
            results.append([prompt, out_str, i['ID'], f"{cache}/mgpt_{prompt_type}_{i['ID']}_prob.json", 'mgpt'])
        else:
            print('error')
    pd.DataFrame(results, columns=['prompt', 'response_str', 'ID', 'prob_addr', 'model']).to_csv(f"cache/mgpt_{prompt_type}.csv")
    return results

import pandas as pd

df = pd.read_csv('MPLU_text.csv')

res = eval_mgpt(df, load_cache=True)