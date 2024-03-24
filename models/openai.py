import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
import torch
import itertools
import json


df = pd.read_csv('MPLU_text.csv')


client = OpenAI(
    api_key="",
)

engin='gpt-4'


def eval_openai(row, prompt_type=0):
    cache_dir = os.listdir('cache')

    if f'{engin}_{prompt_type}_{row['ID']}_prompt.txt' in cache_dir and f'{engin}_{prompt_type}_{row['ID']}_str.txt' in cache_dir:
        with open(os.path.join('cache', f'{engin}_{prompt_type}_{row['ID']}_prompt.txt'), 'r', encoding='utf-8') as f:
            prompt = f.read()
        with open(os.path.join('cache', f'{engin}_{prompt_type}_{row['ID']}_str.txt'), 'r', encoding='utf-8') as f:
            outputs = f.read()
        return [prompt, outputs, row['ID'], engin]

    else:
        prompt = generate_prompt(row)

        try:
            completion = client.chat.completions.create(
                model=engin,
                messages=[
                {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            response = completion.choices[0].message.content
            with open(os.path.join('cache', f'{engin}_{prompt_type}_{row['ID']}_prompt.txt'), 'w', encoding="utf-8") as f:
                f.write(prompt)

            with open(os.path.join('cache', f'{engin}_{prompt_type}_{row['ID']}_str.txt'), 'w', encoding="utf-8") as f:
                f.write(response)
                
            return [prompt, response, row["ID"], engin]
        except Exception as e:
            print(e)
            return [prompt, 'try again', row["ID"], engin]
        

results = []
with tqdm(total=len(df), desc='Processing items') as pbar:
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(eval_openai, row) for index, row in df.iterrows()]
        for future in as_completed(futures):
            results.append(future.result())
            pbar.update(1)  # Update progress bar
            
