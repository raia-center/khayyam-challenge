import re


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
    if prompt_type==2 or prompt_type==3:
        if 'a' in num:
            return 1
        elif 'b' in num:
            return 2
        elif 'c' in num:
            return 3
        elif 'd' in num:
            return 4

def exteract_option(row, prompt_type=0, mode='regex'):
    if mode=='regex':
        if prompt_type==0:
            pattern1 = r'گزینه[^\S]۱|گزینه[^\S]1|گزینه[^\S]۲|گزینه[^\S]2|گزینه[^\S]۳|گزینه[^\S]3|گزینه[^\S]۴|گزینه[^\S]4'
            pattern2 = r'1\)|۱\)|2\)|۲\)|3\)|۳\)|4\)|۴\)'
            patter1_match = re.findall(pattern1, output)
            patter2_match = re.findall(pattern2, output)
            if len(patter1_match)>1:
                return -1
            elif len(patter1_match)==1:
                return exteract_number(patter1_match[0])
            elif len(patter2_match)>1:
                return -1
            elif len(patter2_match)==1:
                return exteract_number(patter2_match[0])
            else:
                return -1
        if prompt_type==1:
            pattern1 = f"{row['label1']}|{row['label2']}|{row['label3']}|{row['label4']}"
            patter1_match = re.findall(pattern1, row['response_str2_gpt4'])
            if len(patter1_match)>1:
                return -1
            elif len(patter1_match)==1:
                if row['label1'] in row['response_str2_gpt4']:
                    return 1
                elif row['label2'] in row['response_str2_gpt4']:
                    return 2
                elif row['label3'] in row['response_str2_gpt4']:
                    return 3
                elif row['label4'] in row['response_str2_gpt4']:
                    return 4
                else:
                    return -1
            else:
                return -1
    elif mode=='option_prob':
        pass
    elif mode=='full_choice_prob':
        pass
    

def eval(model_name_list):
    pass