import json
import random

def set_name_array():
    name_array = []
    with open("yugioh_cards.json", 'r',encoding="utf-8") as f:
        data = json.load(f)
    for card in data['Cards']:
        name_array.append(card["Name"])   
    return name_array

def get_random_name(name_array):
    t = random.randint(0 , len(name_array))
    return str(name_array[t])

'''
Testing
array = set_name_array()
print(get_random_name(array))
'''