from tqdm import tqdm
import random


def eval_random(df, prompt_type=0):
    results = []
    for _, i in tqdm(df.iterrows()):
        out_str = random.choice(['1)', '2)', '3)', '4)'])
        results.append([out_str, i['ID'], 'random'])
    pd.DataFrame(results, columns=['response_str', 'ID', 'model']).to_csv(f"cache/random_{prompt_type}.csv")
    return results

import pandas as pd

df = pd.read_csv('MPLU_text.csv')

res = eval_random(df)