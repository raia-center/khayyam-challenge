import random
import string

def random_str():
  characters = string.ascii_letters + string.digits
  random_string = ''.join(random.choice(characters) for _ in range(6))

  return random_string


def create_choices(list_):
    return list_[0]+", "+list_[1]+", "+list_[2]+", "+list_[3]

def generate_prompt(row, mode=0, CoT=False, just_calibration=False, df_context=None):
    if mode==0:
        persian_number ={
        '1': '۱', '2': '۲', '3': '۳', '4': '۴'
        }

        prompt="در زیر سوالات چند گزینه‌ای (با پاسخ) در مورد " + \
        str(row['final_category_fa']) + \
        " را مشاهده می‌کنید." + \
        "\nسوال: "+ \
        str(row["Question Body"]) +\
        "\nگزینه‌ها:"
        prompt+="\n" + persian_number['1'] + ") " +str(row["Choice 1"])
        prompt+="\n" + persian_number['2'] + ") " +str(row["Choice 2"])
        prompt+="\n" + persian_number['3'] + ") " +str(row["Choice 3"])
        prompt+="\n" + persian_number['4'] + ") " +str(row["Choice 4"])
        prompt+="\nجواب: "
        if CoT:
            prompt+=" بیایید قدم به قدم فکر کنیم"
    elif mode==1:
        pass
        # persian instruction
        # aAn5Uhe
        list_index=[]
        for _ in range(4):
            list_index.append(random_str())
        list_index1=list_index.copy()
        random.shuffle(list_index)
        prompt = "در زیر سوالات چند گزینه‌ای (با پاسخ) در مورد " + \
                str(row['final_category']) + \
                " را مشاهده می‌کنید." + \
                " پاسخ‌ها در بین گزینه‌های " + \
                create_choices(list_index) + \
                ' است.' + \
                "\nسوال: " + \
                str(row["Question Body"]) + \
                "\nگزینه‌ها:"

        random.shuffle(list_index1)
        prompt += "\n" + list_index1[0] + ") " + str(row["Choice 1"])
        prompt += "\n" + list_index1[1] + ") " + str(row["Choice 2"])
        prompt += "\n" + list_index1[2] + ") " + str(row["Choice 3"])
        prompt += "\n" + list_index1[3] + ") " + str(row["Choice 4"])
        prompt += "\nجواب: "
        
    elif mode==2:
        char_map ={
        '1': 'a', '2': 'b', '3': 'c', '4': 'd'
        }

        prompt="در زیر سوالات چند گزینه‌ای (با پاسخ) در مورد " + \
        str(row['final_category_fa']) + \
        " را مشاهده می‌کنید." + \
        "\nسوال: "+ \
        str(row["Question Body"]) +\
        "\nگزینه‌ها:"
        prompt+="\n" + char_map['1'] + ". " +str(row["Choice 1"])
        prompt+="\n" + char_map['2'] + ". " +str(row["Choice 2"])
        prompt+="\n" + char_map['3'] + ". " +str(row["Choice 3"])
        prompt+="\n" + char_map['4'] + ". " +str(row["Choice 4"])
        prompt+="\nجواب: "
        if CoT:
            prompt+=" بیایید قدم به قدم فکر کنیم"
    elif mode==3:
        char_map ={
        '1': 'a', '2': 'b', '3': 'c', '4': 'd'
        }

        prompt="The following are multiple choice questions (with answer) about " + \
        str(row['final_category_fa']) + \
        "." + \
        "\nسوال: "+ \
        str(row["Question Body"]) +\
        "\nگزینه‌ها:"
        prompt+="\n" + char_map['1'] + ". " +str(row["Choice 1"])
        prompt+="\n" + char_map['2'] + ". " +str(row["Choice 2"])
        prompt+="\n" + char_map['3'] + ". " +str(row["Choice 3"])
        prompt+="\n" + char_map['4'] + ". " +str(row["Choice 4"])
        prompt+="\nجواب: "
        if CoT:
            prompt+=" بیایید قدم به قدم فکر کنیم"
    return prompt