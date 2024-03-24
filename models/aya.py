from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
from tqdm import tqdm
import torch
import itertools
import json

checkpoint = "CohereForAI/aya-101"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto")

        
def eval_aya(df, prompt_type=0):
    results = []
    cache_dir = os.listdir('cache')
    for _, i in tqdm(df.iterrows()):
        
        if f'aya_{prompt_type}_{i['ID']}_prompt.txt' in cache_dir and f'aya_{prompt_type}_{i['ID']}_str.txt' in cache_dir and f'aya_{prompt_type}_{i['ID']}_prob.json' in cache_dir:
            with open(os.path.join('cache', f'aya_{prompt_type}_{i['ID']}_prompt.txt'), 'r', encoding='utf-8') as f:
                prompt = f.read()
            with open(os.path.join('cache', f'aya_{prompt_type}_{i['ID']}_str.txt'), 'r', encoding='utf-8') as f:
                outputs = f.read()
            results.append([prompt, outputs, i['ID'], 'aya-101'])

        else:
            prompt = generate_prompt(i)

            inputs = tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to('cuda')
            outputs = aya_model.generate(inputs, temperature=0, max_new_tokens=4096, return_dict_in_generate=True, output_scores=True)
            
            candidate_token = ['1', '2', '3', '4', '۱', '۲', '۳', '۴', 'الف', 'ب', 'ج', 'د', 'a', 'b', 'c', 'd', str(i["Choice 1"]), str(i["Choice 2"]), str(i["Choice 3"]), str(i["Choice 4"])]
            candidate_token_id = []

            for j in candidate_token:
                candidate_token_id.append(tokenizer.encode(j))

            first_token_probs = torch.softmax(outputs.scores[0], dim=-1)
        
            prob_dict = {}
       
            for j in list(set(itertools.chain(*candidate_token_id))):
                prob_dict[tokenizer.decode(j)] = first_token_probs[0][j].item()


            out_str = tokenizer.decode(outputs.sequences[0])
            
            with open(os.path.join('cache', f'aya_{prompt_type}_{i['ID']}_prompt.txt'), 'w', encoding="utf-8") as f:
                f.write(prompt)

            with open(os.path.join('cache', f'aya_{prompt_type}_{i['ID']}_str.txt'), 'w', encoding="utf-8") as f:
                f.write(out_str)

            with open(os.path.join('cache', f'aya_{prompt_type}_{i['ID']}_prob.json'), 'w', encoding='utf-8') as f:
                json.dump(prob_dict, f, ensure_ascii=False, indent=4)

            results.append([prompt, out_str, i['ID'], 'aya-101'])
    return results

import pandas as pd

df = pd.read_csv('MPLU_text.csv')

res = eval_aya(df)