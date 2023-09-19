import pandas as pd
import json
import numpy as np
from collections import Counter

def cosine_sim(a,b):
    """
    :param a: vector 1
    :param b: vector 2
    :return:
    calculate the cosine similarity as a distance or metrics
    to determine closeness between a and b
    """
    return np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))

vector_trait = ["creative", "logical-thinker", "analytical",
                "data-driven", "user-focused", "mathematical-inclined",
                "tech-savvy", "empathetic", "infrastructure"]

vector_value = [0 for i in range(len(vector_trait))]

question     = pd.read_json("../data/questions.json")

with open('../data/answer.json') as f:
   answer = json.load(f)

with open('../data/career_attribute.json') as f:
   career_attribute = json.load(f)

print(career_attribute)

# Initialize an empty result array
career_track_vectorized = []

# Iterate through the data
for entry in career_attribute:
    career = entry['career']
    values = entry['value']

    # Create a vector with all values initialized to 0
    career_vector = [0] * len(vector_trait)

    # Fill in the vector with values from the data
    for i, trait in enumerate(vector_trait):
        if trait in values:
            career_vector[i] = values[trait]

    # Append the tuple (career, career_vector) to the career_track_vectorized array
    career_track_vectorized.append((career, career_vector))

print(career_track_vectorized)

cont_of_questions_answered = []
for q in question["question"]:
    print(q)

    for i, choices in enumerate(answer[q]):
        print(i, choices)

    ans = int(input("answer: "))
    cont_of_questions_answered.append(np.array(list(answer[q].values()))[ans])

# Use Counter to count the occurrences of each element
element_count = Counter(cont_of_questions_answered)

# Convert the Counter object to a dictionary
count_dict = dict(element_count)

for k in count_dict.keys():
    index = vector_trait.index(k)
    vector_value[index] = count_dict[k]


print(vector_trait)
print(vector_value)

job_rec = [(i[0],cosine_sim(i[1],vector_value))
           for i in career_track_vectorized]
# Sort the data by the second element of each tuple in descending order
job_rec = sorted(job_rec, key=lambda x: x[1], reverse=True)

print(job_rec)
if __name__ == "__main__":
    pass