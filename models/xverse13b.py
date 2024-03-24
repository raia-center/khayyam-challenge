# XVERSE.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import os
from tqdm import tqdm
import torch
import itertools
import json

model_path = "xverse/XVERSE-13B-Chat"
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

model = model.eval()

def eval_xverse(df, prompt_type=0):
    results = []
    cache_dir = os.listdir('cache')
    for _, i in tqdm(df.iterrows()):
        
        if f"xverse_{prompt_type}_{i['ID']}_prompt.txt" in cache_dir and f"xverse_{prompt_type}_{i['ID']}_str.txt" in cache_dir and f"xverse_{prompt_type}_{i['ID']}_prob.json" in cache_dir:
            with open(os.path.join('cache', f"xverse_{prompt_type}_{i['ID']}_prompt.txt"), 'r', encoding='utf-8') as f:
                prompt = f.read()
            with open(os.path.join('cache', f"xverse_{prompt_type}_{i['ID']}_str.txt"), 'r', encoding='utf-8') as f:
                outputs = f.read()
            results.append([prompt, outputs, i['ID'], 'xverse'])

        else:
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
            
            with open(os.path.join('cache', f"xverse_{prompt_type}_{i['ID']}_prompt.txt"), 'w', encoding="utf-8") as f:
                f.write(prompt)

            with open(os.path.join('cache', f"xverse_{prompt_type}_{i['ID']}_str.txt"), 'w', encoding="utf-8") as f:
                f.write(out_str)

            with open(os.path.join('cache', f"xverse_{prompt_type}_{i['ID']}_prob.json"), 'w', encoding='utf-8') as f:
                json.dump(prob_dict, f, ensure_ascii=False, indent=4)

            results.append([prompt, out_str, i['ID'], 'xverse'])
    return results

import pandas as pd

df = pd.read_csv('MPLU_text.csv')

res = eval_xverse(df)