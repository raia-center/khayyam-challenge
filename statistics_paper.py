import os
from scipy.spatial import distance
from all_dict import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import os
import numpy as np


def option_table():
    data_df = pd.read_csv('MPLU_text.csv')

    model_name1 = []
    model_name2 = []
    dist_1 = []
    dist_2 = []
    dist_3 = []
    dist_4 = []
    dist_0 = []
    jensenshannon_distance = []

    model_name1.append('Ground Truth')
    model_name2.append('Ground Truth')

    dist_1.append(data_df['Key'].value_counts().to_dict()[1])
    dist_2.append(data_df['Key'].value_counts().to_dict()[2])
    dist_3.append(data_df['Key'].value_counts().to_dict()[3])
    dist_4.append(data_df['Key'].value_counts().to_dict()[4])
    dist_0.append(data_df['Key'].value_counts().to_dict().get(-1, 0))
    jensenshannon_distance.append(distance.jensenshannon([data_df['Key'].value_counts().to_dict()[1],data_df['Key'].value_counts().to_dict()[2],data_df['Key'].value_counts().to_dict()[3],data_df['Key'].value_counts().to_dict()[4]],
                                                        [data_df['Key'].value_counts().to_dict()[1],data_df['Key'].value_counts().to_dict()[2],data_df['Key'].value_counts().to_dict()[3],data_df['Key'].value_counts().to_dict()[4]]))


    for i in tqdm(os.listdir('cache')):
        if 'csv' in i:
            df = pd.read_csv(os.path.join('cache', i))
            df = pd.merge(data_df, df, on=['ID'], how='inner', suffixes=('', '_'+i))
            if i=='gpt4_0.csv' or i=='gpt35_0.csv' or i=='haiku_0.csv':
                cols = ['regex', 'regex_post']
            elif 'random' in i:
                cols = ['regex']
            else:
                cols = ['regex', 'regex_post', 'prob_ans', 'full_prob_ans']
            for j in cols:
                model_name1.append(i)
                model_name2.append(j)
                dist_1.append(df[j].value_counts().to_dict()[1])
                dist_2.append(df[j].value_counts().to_dict()[2])
                dist_3.append(df[j].value_counts().to_dict()[3])
                dist_4.append(df[j].value_counts().to_dict()[4])
                dist_0.append(df[j].value_counts().to_dict().get(-1, 0))
                jensenshannon_distance.append(distance.jensenshannon([df['Key'].value_counts().to_dict()[1],df['Key'].value_counts().to_dict()[2],df['Key'].value_counts().to_dict()[3],df['Key'].value_counts().to_dict()[4]],
                                                                    [df[j].value_counts().to_dict()[1],df[j].value_counts().to_dict()[2],df[j].value_counts().to_dict()[3],df[j].value_counts().to_dict()[4]]))
            

            
    pd.DataFrame({'Model1':model_name1,
                'Model2':model_name2,
                    '1)':dist_1,
                    '2)':dist_2,
                    '3)':dist_3,
                    '4)':dist_4,
                    'No Answer':dist_0,
                    'Distance':jensenshannon_distance}).round(decimals=2).to_csv('res/option_dist.csv')



def create_table(data_df, human_df, fields, file_name, folder='res2', just_regex=False):
    # df = pd.merge(data_df, human_df, on=['ID'], how='inner')
    # y_true = df.groupby(fields)['Key'].apply(list).to_dict()
    # y_pred = df.groupby(fields)['human_answer'].apply(list).to_dict()
    # name1 = []
    # name2 = []
    # acc = []

    # for j in y_true.keys():
    #     name1.append(j[0])
    #     name2.append(j[1])

    #     acc.append(accuracy_score(y_true[j], y_pred[j]))
        
    # name1.append('Avg on all tasks')
    # name2.append('Avg on all tasks')
    # acc.append(np.mean(np.array(acc)))
        
    # name1.append('Avg on all questions')
    # name2.append('Avg on all questions')
    # acc.append(accuracy_score(list(df['Key']), list(df['human_answer'])))

    # human_acc = pd.DataFrame({'name1':name1, 'name2':name2, 'human':acc})

    dfs = []

    for i in tqdm(os.listdir('cache')):
        if 'csv' in i:
            df = pd.read_csv(os.path.join('cache', i))
            df = pd.merge(data_df, df, on=['ID'], how='inner', suffixes=('', '_'+i))
            y_true = df.groupby(fields)['Key'].apply(list).to_dict()
            y_pred = df.groupby(fields)['regex_post'].apply(list).to_dict()
            name1 = []
            name2 = []
            acc = []
            for j in y_true.keys():
                name1.append(j[0])
                name2.append(j[1])
                acc.append(accuracy_score(y_true[j], y_pred[j]))
                
            name1.append('Avg on all tasks')
            name2.append('Avg on all tasks')
            acc.append(np.mean(np.array(acc)))
                
            name1.append('Avg on all questions')
            name2.append('Avg on all questions')
            acc.append(accuracy_score(list(df['Key']), list(df['regex_post'])))
            
            dfs.append(pd.DataFrame({'name1':name1,
                                    'name2':name2,
                                    i[:-6]:acc}))
            
    c = 0
    acc_df = pd.merge(dfs[0], dfs[1], on=['name1', 'name2'], how='inner', suffixes=(str(c), str(c+1)))
    for i in dfs[2:]:
        c+=2
        acc_df = pd.merge(acc_df, i, on=['name1', 'name2'], how='inner', suffixes=(str(c), str(c+1)))
    c+=2
    acc_df = pd.merge(acc_df, human_acc, on=['name1', 'name2'], how='inner', suffixes=(str(c), str(c+1)))
    acc_df.round(decimals=2).to_csv(f'{folder}/{file_name}_regex.csv')
    
    if just_regex:
        return

    dfs = []

    for i in tqdm(os.listdir('cache')):
        if 'csv' in i:
            df = pd.read_csv(os.path.join('cache', i))
            df = pd.merge(data_df, df, on=['ID'], how='inner', suffixes=('', '_'+i))
            y_true = df.groupby(fields)['Key'].apply(list).to_dict()
            
            if i=='gpt4_0.csv' or i=='gpt35_0.csv' or i=='random_0.csv' or i=='haiku_0.csv':
                y_pred = df.groupby(fields)['regex_post'].apply(list).to_dict()
            else:
                y_pred = df.groupby(fields)['prob_ans'].apply(list).to_dict()
                
            name1 = []
            name2 = []
            acc = []
            for j in y_true.keys():
                name1.append(j[0])
                name2.append(j[1])
                acc.append(accuracy_score(y_true[j], y_pred[j]))
                
            name1.append('Avg on all tasks')
            name2.append('Avg on all tasks')
            acc.append(np.mean(np.array(acc)))
                
            name1.append('Avg on all questions')
            name2.append('Avg on all questions')
            if i=='gpt4_0.csv' or i=='gpt35_0.csv' or i=='random_0.csv' or i=='haiku_0.csv':
                acc.append(accuracy_score(list(df['Key']), list(df['regex_post'])))
            else:
                acc.append(accuracy_score(list(df['Key']), list(df['prob_ans'])))
            
            dfs.append(pd.DataFrame({'name1':name1,
                                    'name2':name2,
                                i[:-6]:acc,}))
            
    c = 0
    acc_df = pd.merge(dfs[0], dfs[1], on=['name1', 'name2'], how='inner', suffixes=(str(c), str(c+1)))
    for i in dfs[2:]:
        c+=2
        acc_df = pd.merge(acc_df, i, on=['name1', 'name2'], how='inner', suffixes=(str(c), str(c+1)))
    c+=2
    acc_df = pd.merge(acc_df, human_acc, on=['name1', 'name2'], how='inner', suffixes=(str(c), str(c+1)))
    acc_df.round(decimals=2).to_csv(f'{folder}/{file_name}_prob.csv')
        
    dfs = []

    for i in tqdm(os.listdir('cache')):
        if 'csv' in i:
            df = pd.read_csv(os.path.join('cache', i))
            df = pd.merge(data_df, df, on=['ID'], how='inner', suffixes=('', '_'+i))
            
            y_true = df.groupby(fields)['Key'].apply(list).to_dict()
            
            if i=='gpt4_0.csv' or i=='gpt35_0.csv' or i=='random_0.csv' or i=='haiku_0.csv':
                y_pred = df.groupby(fields)['regex_post'].apply(list).to_dict()
            else:
                y_pred = df.groupby(fields)['full_prob_ans'].apply(list).to_dict()
                
            name1 = []
            name2 = []
            acc = []

            for j in y_true.keys():
                name1.append(j[0])
                name2.append(j[1])
                acc.append(accuracy_score(y_true[j], y_pred[j]))
                
            name1.append('Avg on all tasks')
            name2.append('Avg on all tasks')
            acc.append(np.mean(np.array(acc)))
                
            name1.append('Avg on all questions')
            name2.append('Avg on all questions')
            
            if i=='gpt4_0.csv' or i=='gpt35_0.csv' or i=='random_0.csv' or i=='haiku_0.csv':
                acc.append(accuracy_score(list(df['Key']), list(df['regex_post'])))
            else:
                acc.append(accuracy_score(list(df['Key']), list(df['full_prob_ans'])))
                
            
            dfs.append(pd.DataFrame({'name1':name1,
                                    'name2':name2,
                                i[:-6]:acc,}))
            
    c = 0
    acc_df = pd.merge(dfs[0], dfs[1], on=['name1', 'name2'], how='inner', suffixes=(str(c), str(c+1)))
    for i in dfs[2:]:
        c+=2
        acc_df = pd.merge(acc_df, i, on=['name1', 'name2'], how='inner', suffixes=(str(c), str(c+1)))
        
    c+=2
    acc_df = pd.merge(acc_df, human_acc, on=['name1', 'name2'], how='inner', suffixes=(str(c), str(c+1)))
    acc_df.round(decimals=2).to_csv(f'{folder}/{file_name}_fullprob.csv')
    

human_df = pd.read_csv('human_eval.csv')
data_df = pd.read_csv('MPLU_text.csv')

data_df['unleveled_final_category'] = data_df['final_category_fa'].map(unleveled_final_category) 
data_df['group_unleveledـfinal_category'] = data_df['unleveled_final_category'].map(group_unleveledـfinal_category) 
data_df['group_unleveledـfinal_category_en'] = data_df['group_unleveledـfinal_category'].map(group_unleveledـfinal_category_en) 
data_df['Level_en'] = data_df['Level'].map(level_map)
data_df['Year_en'] = data_df['Year'].map(years_map)
data_df['final_category_en'] = data_df['final_category_fa'].map(final_category_en)
data_df['Education Period_en'] = data_df['Education Period'].map(edu_map)

create_table(data_df, human_df, ['final_category_en', 'final_category_en'], 'acc_all_cat')



human_df = pd.read_csv('human_eval.csv')
data_df = pd.read_csv('MPLU_text.csv')

data_df['unleveled_final_category'] = data_df['final_category_fa'].map(unleveled_final_category) 
data_df['group_unleveledـfinal_category'] = data_df['unleveled_final_category'].map(group_unleveledـfinal_category) 
data_df['group_unleveledـfinal_category_en'] = data_df['group_unleveledـfinal_category'].map(group_unleveledـfinal_category_en) 
data_df['Level_en'] = data_df['Level'].map(level_map)
data_df['Year_en'] = data_df['Year'].map(years_map)
data_df['final_category_en'] = data_df['final_category_fa'].map(final_category_en)
data_df['Education Period_en'] = data_df['Education Period'].map(edu_map)

create_table(data_df, human_df, ['group_unleveledـfinal_category_en', 'group_unleveledـfinal_category_en'], 'acc_merged_cat')


human_df = pd.read_csv('human_eval.csv')
data_df = pd.read_csv('MPLU_text.csv')

data_df['unleveled_final_category'] = data_df['final_category_fa'].map(unleveled_final_category) 
data_df['group_unleveledـfinal_category'] = data_df['unleveled_final_category'].map(group_unleveledـfinal_category) 
data_df['group_unleveledـfinal_category_en'] = data_df['group_unleveledـfinal_category'].map(group_unleveledـfinal_category_en) 
data_df['Level_en'] = data_df['Level'].map(level_map)
data_df['Year_en'] = data_df['Year'].map(years_map)
data_df['final_category_en'] = data_df['final_category_fa'].map(final_category_en)
data_df['Education Period_en'] = data_df['Education Period'].map(edu_map)

create_table(data_df, human_df, ['group_unleveledـfinal_category_en', 'Level_en'], 'acc_lvl_merged_cat')



human_df = pd.read_csv('human_eval.csv')
data_df = pd.read_csv('MPLU_text.csv')

data_df['unleveled_final_category'] = data_df['final_category_fa'].map(unleveled_final_category) 
data_df['group_unleveledـfinal_category'] = data_df['unleveled_final_category'].map(group_unleveledـfinal_category) 
data_df['group_unleveledـfinal_category_en'] = data_df['group_unleveledـfinal_category'].map(group_unleveledـfinal_category_en) 
data_df['Level_en'] = data_df['Level'].map(level_map)
data_df['Year_en'] = data_df['Year'].map(years_map)
data_df['final_category_en'] = data_df['final_category_fa'].map(final_category_en)
data_df['Education Period_en'] = data_df['Education Period'].map(edu_map)

create_table(data_df, human_df, ['group_unleveledـfinal_category_en', 'Year_en'], 'acc_year_merged_cat')



human_df = pd.read_csv('human_eval.csv')
data_df = pd.read_csv('MPLU_text.csv')

data_df['unleveled_final_category'] = data_df['final_category_fa'].map(unleveled_final_category) 
data_df['group_unleveledـfinal_category'] = data_df['unleveled_final_category'].map(group_unleveledـfinal_category) 
data_df['group_unleveledـfinal_category_en'] = data_df['group_unleveledـfinal_category'].map(group_unleveledـfinal_category_en) 
data_df['Level_en'] = data_df['Level'].map(level_map)
data_df['Year_en'] = data_df['Year'].map(years_map)
data_df['final_category_en'] = data_df['final_category_fa'].map(final_category_en)
data_df['Education Period_en'] = data_df['Education Period'].map(edu_map)

create_table(data_df, human_df, ['Year_en', 'Level_en'], 'acc_year_lvl')


human_df = pd.read_csv('human_eval.csv')
data_df = pd.read_csv('MPLU_text.csv')

data_df['unleveled_final_category'] = data_df['final_category_fa'].map(unleveled_final_category) 
data_df['group_unleveledـfinal_category'] = data_df['unleveled_final_category'].map(group_unleveledـfinal_category) 
data_df['group_unleveledـfinal_category_en'] = data_df['group_unleveledـfinal_category'].map(group_unleveledـfinal_category_en) 
data_df['Level_en'] = data_df['Level'].map(level_map)
data_df['Year_en'] = data_df['Year'].map(years_map)
data_df['final_category_en'] = data_df['final_category_fa'].map(final_category_en)
data_df['Education Period_en'] = data_df['Education Period'].map(edu_map)

create_table(data_df, human_df, ['Education Period_en', 'Level_en'], 'acc_edu_lvl')


from scipy import stats
import pandas as pd
from tqdm import tqdm
import os


human_df = pd.read_csv('human_eval.csv')
data_df = pd.read_csv('MPLU_text.csv')

data_df['unleveled_final_category'] = data_df['final_category_fa'].map(unleveled_final_category) 
data_df['group_unleveledـfinal_category'] = data_df['unleveled_final_category'].map(group_unleveledـfinal_category) 
data_df['group_unleveledـfinal_category_en'] = data_df['group_unleveledـfinal_category'].map(group_unleveledـfinal_category_en) 
data_df['Level_en'] = data_df['Level'].map(level_map)
data_df['Year_en'] = data_df['Year'].map(years_map)
data_df['final_category_en'] = data_df['final_category_fa'].map(final_category_en)
data_df['Education Period_en'] = data_df['Education Period'].map(edu_map)


dfs = []
c = 0
for k in ['response_str', 'Question Body', 'Answer']:
    df_ttest = {}
    for i in tqdm(os.listdir('cache')):
        if 'csv' in i and 'random' not in i:
            c+=1
            df = pd.read_csv(os.path.join('cache', i))
            df = pd.merge(data_df, df, on=['ID'], how='inner', suffixes=('', '_'+i))
            
            list1 = []
            list2 = []
                
            for index, row in df.iterrows():
                if int(row['Key'])==int(row['regex_post']):
                    list1.append(len(str(row[k])))
                else:
                    list2.append(len(str(row[k])))
            stat = stats.ttest_ind(list1, list2)
            df_ttest[c] = [i[:-6], str(stat.statistic.round(decimals=2)) + ' / '+ str(stat.pvalue.round(decimals=2))]
            
    dfs.append(pd.DataFrame.from_dict(df_ttest, orient='index', columns=['Model', 'T-Test']))
            

ttest_df = pd.merge(dfs[0], dfs[1], on=['Model'], how='inner', suffixes=('_response_str', '_Question Body'))
ttest_df = pd.merge(ttest_df, dfs[2], on=['Model'], how='inner', suffixes=('', '_Answer'))


ttest_df.round(decimals=2).to_csv('res2/ttest.csv')



from scipy import stats
import pandas as pd
import os


human_df = pd.read_csv('human_eval.csv')
data_df = pd.read_csv('MPLU_text.csv')

data_df['unleveled_final_category'] = data_df['final_category_fa'].map(unleveled_final_category) 
data_df['group_unleveledـfinal_category'] = data_df['unleveled_final_category'].map(group_unleveledـfinal_category) 
data_df['group_unleveledـfinal_category_en'] = data_df['group_unleveledـfinal_category'].map(group_unleveledـfinal_category_en) 
data_df['Level_en'] = data_df['Level'].map(level_map)
data_df['Year_en'] = data_df['Year'].map(years_map)
data_df['final_category_en'] = data_df['final_category_fa'].map(final_category_en)
data_df['Education Period_en'] = data_df['Education Period'].map(edu_map)


for k in ['Question Body', 'Answer']:
    df = pd.merge(data_df, human_df, on=['ID'], how='inner')
    
    list1 = []
    list2 = []
        
    for index, row in df.iterrows():
        if int(row['Key'])==int(row['human_answer']):
            list1.append(len(str(row[k])))
        else:
            list2.append(len(str(row[k])))
            
    print(k, ': ', stats.ttest_ind(list1, list2))
    
    
from statistics_paper import create_table

human_df = pd.read_csv('human_eval.csv')
data_df = pd.read_csv('MPLU_text.csv')

data_df = data_df[data_df['Is Trap']]

data_df['unleveled_final_category'] = data_df['final_category_fa'].map(unleveled_final_category) 
data_df['group_unleveledـfinal_category'] = data_df['unleveled_final_category'].map(group_unleveledـfinal_category) 
data_df['group_unleveledـfinal_category_en'] = data_df['group_unleveledـfinal_category'].map(group_unleveledـfinal_category_en) 
data_df['Level_en'] = data_df['Level'].map(level_map)
data_df['Year_en'] = data_df['Year'].map(years_map)
data_df['final_category_en'] = data_df['final_category_fa'].map(final_category_en)
data_df['Education Period_en'] = data_df['Education Period'].map(edu_map)

create_table(data_df, human_df, ['group_unleveledـfinal_category_en', 'group_unleveledـfinal_category_en'], 'acc_trap_all_cat', just_regex=True)

df1 = pd.read_csv('res2/acc_trap_all_cat_regex.csv')
df2 = pd.read_csv('res2/acc_merged_cat_regex.csv')
df = pd.merge(df1, df2, on=['name1', 'name2'], how='inner', suffixes=('_trap', ''))

trap_df = pd.DataFrame()
trap_df['name'] = df['name1']
for i in df1.columns:
    if i!='name1' and i!='name2':
        trap_df[i] = df[i].astype(str) + ' / ' + df[i+'_trap'].astype(str)
trap_df.to_csv('res2/acc_trap.csv')