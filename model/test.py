from model.data_interface.data import *
from model.inference.training import *
from model.inference.recommendation import *
from surprise import dump

import random

my_seed = 42
random.seed(my_seed)


data = load_data(data_path="../data/dicoding_user_item_rating.gzip")
print(data.head())

data_input = data.loc[:, ["user_id", "course_id", "rating"]]
surprise_input = reader_data(data=data_input,
                             cols=["user_id", "course_id", "rating"],
                             scale=True,
                             model="surprise")
print((type(surprise_input)))

trainset, testset = surprise_split_train_test(surprise_data = surprise_input)


print(trainset)

# getting the property of user with id of == n
print([user for user in testset if user[0]==623699])

# model_searched = model_search(data=surprise_input, cv=3)
# print(model_searched)
# print("Loading new model for prediction")
# model_searched.to_csv("../backend/model.csv")
#
# params = fine_tuned_model(surprise_input, model="SVD", cv=3)
# print("Attempting to export the model...")
# print(params.best_params['rmse'])
#
# algo = SVDpp(**params.best_params['rmse'])
# algo.fit(trainset)
# dump.dump(file_name="../backend/model/model.pickle", algo=algo)

model_filename = "../backend/model/model.pickle"
model = load_model(model_filename)

pred = model.test([i for i in testset if i[0] == 623699])

top_n = get_top_n(pred, n=10)
top_n = list(top_n.items())[0]

print("ID item for given user id")
arr_item_id = [i[0] for i in top_n[1]]
print("rating of items for respectively item id")
arr_rating  = [i[1] for i in top_n[1]]


print(arr_item_id, type(arr_item_id))
print(arr_rating, type(arr_item_id))


for item in arr_item_id:
    ret = item_rating(model, user = 623699, item = item)
    print(ret["rating"])

