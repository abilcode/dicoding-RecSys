import pandas as pd
import json
import numpy as np
import random
from collections import Counter




vector_trait = ["creative", "logical thinker", "analytical", "data-driven", "user-focused", "tech-savvy"]
vector_value = [0 for i in range(10)]

question     = pd.read_json("../data/questions.json")
with open('../data/answer.json') as f:
   answer = json.load(f)

cont = []
for q in question["question"]:
    print(q)

    for i, choices in enumerate(answer[q]):
        print(i, choices)

    ans = int(input("answer: "))
    cont.append(np.array(list(answer[q].values()))[ans])

# Use Counter to count the occurrences of each element
element_count = Counter(cont)

# Convert the Counter object to a dictionary
count_dict = dict(element_count)

for k in count_dict.keys():
    index = vector_trait.index(k)
    vector_value[index] = count_dict[k]

print(vector_value)
if __name__ == "__main__":
    pass