import re
from sklearn.metrics import accuracy_score
from all_dict import *
import random


def exteract_number(num, prompt_type=0):
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

def exteract_option(output, prompt_type=0, mode='regex'):
    if mode=='regex':
        if prompt_type==0:
            pattern1 = r'گزینه[^\S]۱|گزینه[^\S]1|گزینه[^\S]۲|گزینه[^\S]2|گزینه[^\S]۳|گزینه[^\S]3|گزینه[^\S]۴|گزینه[^\S]4'
            pattern2 = r'1\)|۱\)|2\)|۲\)|3\)|۳\)|4\)|۴\)'
            patter1_match = re.findall(pattern1, output)
            patter2_match = re.findall(pattern2, output)
            if len(patter1_match)>1:
                return random.choice([1,2,3,4])
            elif len(patter1_match)==1:
                return exteract_number(patter1_match[0])
            elif len(patter2_match)>1:
                return random.choice([1,2,3,4])
            elif len(patter2_match)==1:
                return exteract_number(patter2_match[0])
            else:
                return random.choice([1,2,3,4])
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
    elif mode=='option_prob':
        pass
    elif mode=='full_choice_prob':
        pass
    


import pandas as pd
import os




data_df = pd.read_csv('MPLU_text.csv')
dfs = []
for i in os.listdir('cache'):
    if 'csv' in i:
        df = pd.read_csv(os.path.join('cache', i))
        df['regex'] = df['response_str'].apply(exteract_option)
        df.to_csv(os.path.join('cache', i))
        
for i in os.listdir('cache'):
    if 'csv' in i:
        df = pd.read_csv(os.path.join('cache', i))
        # suff = i[:-4]
        df = pd.merge(data_df, df, on=['ID'], how='inner', suffixes=('', i[:-4]))
        # df['regex'] = df['response_str'].apply(exteract_option)
        # df.to_csv(os.path.join('cache', i))
        y_true = df.groupby(['final_category_fa'])['Key'].apply(list).to_dict()
        y_pred = df.groupby(['final_category_fa'])['regex'].apply(list).to_dict()


        name = []
        acc = []
        c = []
        for j in y_true.keys():
            name.append(j)
            c.append(len(y_true[j]))
            acc.append(accuracy_score(y_true[j], y_pred[j]))
            # print('-------')
        dfs.append(pd.DataFrame({'name':name,
                            'acc'+i[:-4]:acc,
                            'count':c}))
pd.concat(dfs, axis=1).to_csv('acc_res.csv')
