from model.data_interface.data import *
from model.inference.training import *
from model.inference.recommendation import *
from surprise import dump

data = load_data(data_path="../data/dicoding_user_item_rating.gzip")
print(data.head())

data_input = data.loc[:, ["user_id", "course_id", "rating"]]
surprise_input = reader_data(data=data_input,
                             cols=["user_id", "course_id", "rating"],
                             scale=True,
                             model="surprise")

print((type(surprise_input)))
model_searched = model_search(data=surprise_input, cv=3)
print(model_searched)
model_searched.to_csv("../backend/model.csv")

params = fine_tuned_model(surprise_input, model="SVDpp", cv=5)

print("Attempting to export the model...")

print(params.best_params['rmse'])

algo = SVDpp(**params.best_params['rmse'])
trainset = surprise_input.build_full_trainset()
algo.fit(trainset)
dump.dump(file_name="../backend/model/model.pickle", algo=algo)

model_filename = "../backend/model/model.pickle"
model = load_model(model_filename)

cont_rec = []

for i in data.course_id.unique():
    rating = item_rating(model, user=data.iloc[100,0], item=f"{i}")['rating']
    cont_rec.append((i, rating))

recommendations = sorted(cont_rec, key=lambda x: x[1], reverse=True)

print(recommendations)
