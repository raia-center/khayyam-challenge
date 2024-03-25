from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
from tqdm import tqdm
import torch
import itertools
import json
from prompt import *

        
def eval_mt0(df, prompt_type=0, load_cache=False):
    if not load_cache:
        checkpoint = "bigscience/mt0-xl"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

    results = []
    cache = f'cache/mt0xl_{prompt_type}'
    cache_dir = os.listdir(cache)
    
    for _, i in tqdm(df.iterrows()):
        
        if f"mt0_{prompt_type}_{i['ID']}_prompt.txt" in cache_dir and f"mt0_{prompt_type}_{i['ID']}_str.txt" in cache_dir and f"mt0_{prompt_type}_{i['ID']}_prob.json" in cache_dir:
            with open(os.path.join(cache, f"mt0_{prompt_type}_{i['ID']}_prompt.txt"), 'r', encoding='utf-8') as f:
                prompt = f.read()
            with open(os.path.join(cache, f"mt0_{prompt_type}_{i['ID']}_str.txt"), 'r', encoding='utf-8') as f:
                outputs = f.read()
            results.append([prompt, outputs, i['ID'], f"{cache}/mt0_{prompt_type}_{i['ID']}_prob.json", 'mt0'])

        elif not load_cache:

            prompt = generate_prompt(i)

            inputs = tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to('cuda')
            outputs = model.generate(inputs, temperature=0, max_new_tokens=2048, return_dict_in_generate=True, output_scores=True)
            
            candidate_token = ['1', '2', '3', '4', '۱', '۲', '۳', '۴', 'الف', 'ب', 'ج', 'د', 'a', 'b', 'c', 'd', str(i["Choice 1"]), str(i["Choice 2"]), str(i["Choice 3"]), str(i["Choice 4"])]
            candidate_token_id = []

            for j in candidate_token:
                candidate_token_id.append(tokenizer.encode(j))

            first_token_probs = torch.softmax(outputs.scores[0], dim=-1)
        
            prob_dict = {}
       
            for j in list(set(itertools.chain(*candidate_token_id))):
                prob_dict[tokenizer.decode(j)] = first_token_probs[0][j].item()


            out_str = tokenizer.decode(outputs.sequences[0])
            
            with open(os.path.join(cache, f"mt0_{prompt_type}_{i['ID']}_prompt.txt"), 'w', encoding="utf-8") as f:
                f.write(prompt)

            with open(os.path.join(cache, f"mt0_{prompt_type}_{i['ID']}_str.txt"), 'w', encoding="utf-8") as f:
                f.write(out_str)

            with open(os.path.join(cache, f"mt0_{prompt_type}_{i['ID']}_prob.json"), 'w', encoding='utf-8') as f:
                json.dump(prob_dict, f, ensure_ascii=False, indent=4)
            results.append([prompt, out_str, i['ID'],f"{cache}/mt0_{prompt_type}_{i['ID']}_prob.json", 'mt0'])
        else:
            print('error')
    pd.DataFrame(results, columns=['prompt', 'response_str', 'ID', 'prob_addr', 'model']).to_csv(f"cache/mt0xl_{prompt_type}.csv")

    return results


import pandas as pd

df = pd.read_csv('MPLU_text.csv')

res = eval_mt0(df, load_cache=True)