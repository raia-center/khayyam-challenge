import re
import numpy as np
from sklearn.metrics import accuracy_score
from all_dict import *
import random
from tqdm import tqdm
import pandas as pd
import os
import requests
import time
	

def extract_number(num, prompt_type=0):
    if prompt_type==0:
        if '1' in num or '۱' in num:
            return 1
        elif '2' in num or '۲' in num:
            return 2
        elif '3' in num or '۳' in num:
            return 3
        elif '4' in num or '۴' in num:
            return 4
    # if prompt_type==2 or prompt_type==3:
    #     if 'a' in num:
    #         return 1
    #     elif 'b' in num:
    #         return 2
    #     elif 'c' in num:
    #         return 3
    #     elif 'd' in num:
    #         return 4
    
def extract_regex_option_persianmind(row):
    if row['response_str']!=row['response_str']:
        return random.choice([1, 2, 3, 4])
    pattern1 = r'گزینه[\s]*۱|گزینه[\s]*1|گزینه[\s]*۲|گزینه[\s]*2|گزینه[\s]*۳|گزینه[\s]*3|گزینه[\s]*۴|گزینه[\s]*4'
    pattern2 = r'1[\s]*\)|۱[\s]*\)|2[\s]*\)|۲[\s]*\)|3[\s]*\)|۳[\s]*\)|4[\s]*\)|۴[\s]*\)'
    pattern3 = r'PersianMind:[\s]*1[\s]*$|PersianMind:[\s]*2[\s]*$|PersianMind:[\s]*3[\s]*$|PersianMind:[\s]*4[\s]*$|PersianMind:[\s]*۱[\s]*$|PersianMind:[\s]*۲[\s]*$|PersianMind:[\s]*۳[\s]*$|PersianMind:[\s]*۴[\s]*$|پاسخ:[\s]*1[\s]*$|پاسخ:[\s]*2[\s]*$|پاسخ:[\s]*3[\s]*$|پاسخ:[\s]*4[\s]*$|پاسخ:[\s]*۱[\s]*$|پاسخ:[\s]*۲[\s]*$|پاسخ:[\s]*۳[\s]*$|پاسخ:[\s]*۴[\s]*$'
    pattern4 = r':[\s]*1|:[\s]*2|:[\s]*3|:[\s]*4|:[\s]*۱|:[\s]*۲|:[\s]*۳|:[\s]*۴'
    pattern5 = r'صحیح[\s]*1[\s]*است|صحیح[\s]*2[\s]*است|صحیح[\s]*3[\s]*است|صحیح[\s]*4[\s]*است|صحیح[\s]*۱[\s]*است|صحیح[\s]*۲[\s]*است|صحیح[\s]*۳[\s]*است|صحیح[\s]*۴[\s]*است'
    pattern1_match = re.findall(pattern1, row['response_str'])
    pattern2_match = re.findall(pattern2, row['response_str'])
    pattern3_match = re.findall(pattern3, row['response_str'])
    pattern4_match = re.findall(pattern4, row['response_str'])
    pattern5_match = re.findall(pattern5, row['response_str'])

    
    if len(pattern1_match)>1:
        return -1
    elif len(pattern1_match)==1:
        return extract_number(pattern1_match[0])
    elif len(pattern2_match)>1:
        return -1
    elif len(pattern2_match)==1:
        return extract_number(pattern2_match[0])
    elif len(pattern3_match)==1:
        return extract_number(pattern3_match[0])
    elif len(pattern5_match)==1:
        return extract_number(pattern5_match[0])
    elif len(pattern4_match)==1:
        return extract_number(pattern4_match[0])
    else:
        return -1
    
    
def extract_regex_option_mt0(row):
    if row['response_str']!=row['response_str']:
        return random.choice([1, 2, 3, 4])
    pattern1 = r'گزینه[\s]*۱|گزینه[\s]*1|گزینه[\s]*۲|گزینه[\s]*2|گزینه[\s]*۳|گزینه[\s]*3|گزینه[\s]*۴|گزینه[\s]*4'
    pattern2 = r'1[\s]*\)|۱[\s]*\)|2[\s]*\)|۲[\s]*\)|3[\s]*\)|۳[\s]*\)|4[\s]*\)|۴[\s]*\)'
    pattern3 = r'<pad>[\s]*1[\s]*<\/s>|<pad>[\s]*2[\s]*<\/s>|<pad>[\s]*3[\s]*<\/s>|<pad>[\s]*4[\s]*<\/s>|<pad>[\s]*۱[\s]*<\/s>|<pad>[\s]*۲[\s]*<\/s>|<pad>[\s]*۳[\s]*<\/s>|<pad>[\s]*۴[\s]*<\/s>'
    pattern1_match = re.findall(pattern1, row['response_str'])
    pattern2_match = re.findall(pattern2, row['response_str'])
    pattern3_match = re.findall(pattern3, row['response_str'])
    
    if len(pattern3_match)==1:
        return extract_number(pattern3_match[0])
    
    if len(pattern1_match)>1:
        return -1
    elif len(pattern1_match)==1:
        return extract_number(pattern1_match[0])
    elif len(pattern2_match)>1:
        return -1
    elif len(pattern2_match)==1:
        return extract_number(pattern2_match[0])
    else:
        return -1
    
    
def extract_regex_option_xverse(row, prompt_type=0):
    if prompt_type==0:
        if row['response_str']!=row['response_str']:
            return random.choice([1, 2, 3, 4])
        pattern1 = r'گزینه[\s]*۱|گزینه[\s]*1|گزینه[\s]*۲|گزینه[\s]*2|گزینه[\s]*۳|گزینه[\s]*3|گزینه[\s]*۴|گزینه[\s]*4'
        pattern2 = r'1[\s]*\)|۱[\s]*\)|2[\s]*\)|۲[\s]*\)|3[\s]*\)|۳[\s]*\)|4[\s]*\)|۴[\s]*\)'
        pattern3 = r'^1[\s]*<\|endoftext\|>|^2[\s]*<\|endoftext\|>|^3[\s]*<\|endoftext\|>|^4[\s]*<\|endoftext\|>|^۱[\s]*<\|endoftext\|>|^۲[\s]*<\|endoftext\|>|^۳[\s]*<\|endoftext\|>|^۴[\s]*<\|endoftext\|>'
        pattern1_match = re.findall(pattern1, row['response_str'])
        pattern2_match = re.findall(pattern2, row['response_str'])
        pattern3_match = re.findall(pattern3, row['response_str'].lstrip())
        if len(pattern1_match)>1:
            return -1
        elif len(pattern1_match)==1:
            return extract_number(pattern1_match[0])
        elif len(pattern2_match)>1:
            return -1
        elif len(pattern2_match)==1:
            return extract_number(pattern2_match[0])
        elif len(pattern3_match)==1:
            return extract_number(pattern3_match[0])
        else:
            return -1
        
        
def extract_regex_option_mgpt(row):
    if row['response_str']!=row['response_str']:
        return random.choice([1, 2, 3, 4])
    pattern1 = r'گزینه[\s]*۱|گزینه[\s]*1|گزینه[\s]*۲|گزینه[\s]*2|گزینه[\s]*۳|گزینه[\s]*3|گزینه[\s]*۴|گزینه[\s]*4'
    pattern2 = r'1[\s]*\)|۱[\s]*\)|2[\s]*\)|۲[\s]*\)|3[\s]*\)|۳[\s]*\)|4[\s]*\)|۴[\s]*\)'
    pattern1_match = re.findall(pattern1, row['response_str'])
    pattern2_match = re.findall(pattern2, row['response_str'])
    if len(pattern1_match)>1:
        return random.choice([1, 2, 3, 4])
    elif len(pattern1_match)==1:
        return extract_number(pattern1_match[0])
    elif len(pattern2_match)>1:
        return random.choice([1, 2, 3, 4])
    elif len(pattern2_match)==1:
        return extract_number(pattern2_match[0])
    else:
        return random.choice([1, 2, 3, 4])
    

def extract_option(row, prompt_type=0, mode='regex'):
    if mode=='regex':
        if prompt_type==0:
            if row['response_str']!=row['response_str']:
                return random.choice([1, 2, 3, 4])
            pattern1 = r'گزینه[\s]*۱|گزینه[\s]*1|گزینه[\s]*۲|گزینه[\s]*2|گزینه[\s]*۳|گزینه[\s]*3|گزینه[\s]*۴|گزینه[\s]*4'
            pattern2 = r'1[\s]*\)|۱[\s]*\)|2[\s]*\)|۲[\s]*\)|3[\s]*\)|۳[\s]*\)|4[\s]*\)|۴[\s]*\)'
            pattern1_match = re.findall(pattern1, row['response_str'])
            pattern2_match = re.findall(pattern2, row['response_str'])
            if len(pattern1_match)>1:
                return -1
            elif len(pattern1_match)==1:
                return extract_number(pattern1_match[0])
            elif len(pattern2_match)>1:
                return -1
            elif len(pattern2_match)==1:
                return extract_number(pattern2_match[0])
            else:
                return -1
        # if prompt_type==1:
        #     pattern1 = f"{row['label1']}|{row['label2']}|{row['label3']}|{row['label4']}"
        #     patter1_match = re.findall(pattern1, row['response_str2_gpt4'])
        #     if len(patter1_match)>1:
        #         return -1
        #     elif len(patter1_match)==1:
        #         if row['label1'] in row['response_str2_gpt4']:
        #             return 1
        #         elif row['label2'] in row['response_str2_gpt4']:
        #             return 2
        #         elif row['label3'] in row['response_str2_gpt4']:
        #             return 3
        #         elif row['label4'] in row['response_str2_gpt4']:
        #             return 4
        #         else:
        #             return -1
        #     else:
        #         return -1


data_df = pd.read_csv('MPLU_text.csv')

for i in tqdm(os.listdir('cache')):
    if 'csv' in i:
        if 'mt0' in i:
            df = pd.read_csv(os.path.join('cache', i))
            df['regex'] = df.apply(extract_regex_option_mt0, axis=1)
            df.to_csv(os.path.join('cache', i), index=False)
        elif 'persianmind' in i:
            df = pd.read_csv(os.path.join('cache', i))
            df['regex'] = df.apply(extract_regex_option_persianmind, axis=1)
            df.to_csv(os.path.join('cache', i), index=False)
        elif 'xverse' in i:
            df = pd.read_csv(os.path.join('cache', i))
            df['regex'] = df.apply(extract_regex_option_xverse, axis=1)
            df.to_csv(os.path.join('cache', i), index=False)
        elif 'mgpt' in i:
            df = pd.read_csv(os.path.join('cache', i))
            df['regex'] = df.apply(extract_regex_option_mgpt, axis=1)
            df.to_csv(os.path.join('cache', i), index=False)
        else:
            df = pd.read_csv(os.path.join('cache', i))
            df['regex'] = df.apply(extract_option, axis=1)
            df.to_csv(os.path.join('cache', i), index=False)

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from numpy import dot
from sentence_transformers import SentenceTransformer


use_sim_emb=False

if use_sim_emb:


    labseـmodel = SentenceTransformer('sentence-transformers/LaBSE')
    labseـmodel.to('cuda')

    def query_labse(row):
        if 'regex_post' in row.index.to_list():
            return row['regex_post']
        
        if str(int(row['regex']))=='-1':
            embeddings1 = labseـmodel.encode(str(row['response_str']))
            embeddings2 = labseـmodel.encode([str(row['Choice 1']), str(row['Choice 2']), str(row['Choice 3']), str(row['Choice 4'])])
            return np.argmax(dot(embeddings1, embeddings2.T))+1
        else:
            return row['regex']
        
        
    for i in tqdm(os.listdir('cache')):
        if 'csv' in i:
            df = pd.read_csv(os.path.join('cache', i))
            df['regex_post'] = pd.merge(df, data_df, on=['ID'], how='inner', suffixes=('_'+i, '')).apply(query_labse, axis=1)
            df.to_csv(os.path.join('cache', i), index=False)
        
import json

def extract_prob_option(row):
    with open(row['prob_addr'], 'r') as f:
        prob = json.load(f)
    try:
        prob = np.array([prob['۱'], prob['۲'], prob['۳'], prob['۴']])
    except:
        print('Empty_str')
        return random.choice([1, 2, 3, 4])
    return np.argmax(prob)+1

for i in tqdm(os.listdir('cache')):
    if 'csv' in i:
        if 'gpt' not in i and 'random' not in i:
            df = pd.read_csv(os.path.join('cache', i))
            df['prob_ans'] = df.apply(extract_prob_option, axis=1)
            df.to_csv(os.path.join('cache', i), index=False)

from transformers import AutoTokenizer, GPT2Tokenizer
import numpy as np
from all_dict import *
import random
from tqdm import tqdm
import pandas as pd
import os
import json

tqdm.pandas()


def extract_full_prob_option(row, tokenizer):
    with open(row['prob_addr'], 'r') as f:
        prob = json.load(f)
    try:  
        choice1 = tokenizer.encode(row['Choice 1'])
        choice1_prob = 0

        for i in choice1:
            choice1_prob+=np.log(prob[tokenizer.decode(i)]+0.000001)
        choice1_prob+=np.log(prob['۱']+0.000001)
        choice1_prob = choice1_prob/(len(choice1)+1)
        
        choice2 = tokenizer.encode(row['Choice 2'])
        choice2_prob = 0

        for i in choice2:
            choice2_prob+=np.log(prob[tokenizer.decode(i)]+0.000001)
        choice2_prob+=np.log(prob['۲']+0.000001)
        choice2_prob = choice2_prob/(len(choice2)+1)
        
        choice3 = tokenizer.encode(row['Choice 3'])
        choice3_prob = 0

        for i in choice3:
            choice3_prob+=np.log(prob[tokenizer.decode(i)]+0.000001)
        choice3_prob+=np.log(prob['۳']+0.000001)
        choice3_prob = choice3_prob/(len(choice3)+1)
        
        choice4 = tokenizer.encode(row['Choice 4'])
        choice4_prob = 0

        for i in choice4:
            choice4_prob+=np.log(prob[tokenizer.decode(i)]+0.000001)
        choice4_prob+=np.log(prob['۴']+0.000001)
        choice4_prob = choice4_prob/(len(choice4)+1)
        
        prob = np.array([choice1_prob, choice2_prob, choice3_prob, choice4_prob])
    except:
        print('Empty_str')
        return random.choice([1, 2, 3, 4])
    return np.argmax(prob)+1

data_df = pd.read_csv('MPLU_text.csv')

for i in tqdm(os.listdir('cache')):
    if 'csv' in i:
        if 'gpt' not in i and 'random' not in i:
            if 'aya' in i:
                tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-101")
            elif 'mgpt' in i:
                tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/mGPT")
            elif 'mt0' in i:
                tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-xl")
            elif 'persianmind' in i:
                tokenizer = AutoTokenizer.from_pretrained(
                    "universitytehran/PersianMind-v1.0",
                )
            elif 'xverse' in i:
                tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-7B-Chat")
            
            df = pd.read_csv(os.path.join('cache', i))
            df['full_prob_ans'] =  pd.merge(df, data_df, on=['ID'], how='inner', suffixes=('_'+i, '')).progress_apply(extract_full_prob_option, axis=1, tokenizer=tokenizer)
            df.to_csv(os.path.join('cache', i), index=False)
            
            
data_df = pd.read_csv('MPLU_text.csv')

key_map = {'1':'P1', '2':'P2', '3':'P3', '4':'P4'}

def human_eval(i):
    if i['P0']==i['P0']:
        if i[key_map[str(int(i['Key']))]]>=(i['P1']+i['P2']+i['P3']+i['P4']-i[key_map[str(int(i['Key']))]]):
            return i['Key']
        else:
            return -2
    else:
        return -1
    
data_df['human_answer'] = data_df.apply(human_eval, axis=1)
data_df[data_df['human_answer']!=-1.0][['ID','human_answer']].to_csv('human_eval.csv')
        