import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import os
from tqdm import tqdm
import itertools
import json
from prompt import *


def post_process(row):
    return ''.join(row.split('جواب:')[1:])


def eval_xverse(df, prompt_type=0, load_cache=False):
    if not load_cache:
        model_path = "xverse/XVERSE-7B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

        model.generation_config = GenerationConfig(pad_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=2048,
        temperature=0.0001,
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.1,
        do_sample=True,
        transformers_versio="4.29.1")

    results = []
    cache = f'cache/xverse7b_{prompt_type}'
    cache_dir = os.listdir(cache)
    
    for _, i in tqdm(df.iterrows()):
        
        if f"xverse_{prompt_type}_{i['ID']}_prompt.txt" in cache_dir and f"xverse_{prompt_type}_{i['ID']}_str.txt" in cache_dir and f"xverse_{prompt_type}_{i['ID']}_prob.json" in cache_dir:
            with open(os.path.join(cache, f"xverse_{prompt_type}_{i['ID']}_prompt.txt"), 'r', encoding='utf-8') as f:
                prompt = f.read()
            with open(os.path.join(cache, f"xverse_{prompt_type}_{i['ID']}_str.txt"), 'r', encoding='utf-8') as f:
                outputs = f.read()
            outputs = post_process(outputs)
            results.append([prompt, outputs, i['ID'], f"{cache}/xverse_{prompt_type}_{i['ID']}_prob.json", 'xverse7b'])

        elif not load_cache:
            prompt = generate_prompt(i)

            inputs = tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to('cuda')
            outputs = model.generate(inputs, return_dict_in_generate=True, output_scores=True)
            
            candidate_token = ['1', '2', '3', '4', '۱', '۲', '۳', '۴', 'الف', 'ب', 'ج', 'د', 'a', 'b', 'c', 'd', str(i["Choice 1"]), str(i["Choice 2"]), str(i["Choice 3"]), str(i["Choice 4"])]
            candidate_token_id = []

            for j in candidate_token:
                candidate_token_id.append(tokenizer.encode(j))

            first_token_probs = torch.softmax(outputs.scores[0], dim=-1)
        
            prob_dict = {}
       
            for j in list(set(itertools.chain(*candidate_token_id))):
                prob_dict[tokenizer.decode(j)] = first_token_probs[0][j].item()


            out_str = tokenizer.decode(outputs.sequences[0])
            
            with open(os.path.join(cache, f"xverse_{prompt_type}_{i['ID']}_prompt.txt"), 'w', encoding="utf-8") as f:
                f.write(prompt)

            with open(os.path.join(cache, f"xverse_{prompt_type}_{i['ID']}_str.txt"), 'w', encoding="utf-8") as f:
                f.write(out_str)

            with open(os.path.join(cache, f"xverse_{prompt_type}_{i['ID']}_prob.json"), 'w', encoding='utf-8') as f:
                json.dump(prob_dict, f, ensure_ascii=False, indent=4)

            out_str = post_process(out_str)
            results.append([prompt, out_str, i['ID'], f"{cache}/xverse_{prompt_type}_{i['ID']}_prob.json", 'xverse7b'])
        else:
            print('error')
    pd.DataFrame(results, columns=['prompt', 'response_str', 'ID', 'prob_addr', 'model']).to_csv(f"cache/xverse7b_{prompt_type}.csv")

    return results

import pandas as pd

df = pd.read_csv('MPLU_text.csv')

res = eval_xverse(df, load_cache=True)