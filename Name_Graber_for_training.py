import random

def set_name_array():
    name_array = []
    with open("yugioh_cards.txt", 'r',encoding="utf-8") as f:
        for line in f:
            y = line.split("|")
            name_array.append(y[0])
    return name_array

def get_random_name(name_array):
    t = random.randint(0 , len(name_array))
    return str(name_array[t])

'''
Testing
array = set_name_array()
print(get_random_name(array))
'''