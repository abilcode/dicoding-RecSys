from model.data_interface.data import *
from model.inference.training import *
from surprise import dump


data = load_data(data_path="../data/dicoding_user_item_rating.gzip")
print(data.head())


data_input = data.loc[:,["user_id", "course_id", "rating"]]
surprise_input = reader_data(data=data_input,
                             cols=["user_id", "course_id", "rating"],
                             scale=True,
                             model = "surprise")


print((type(surprise_input)))
model_searched = model_search(data = surprise_input, cv = 3)
print(model_searched)
model_searched.to_csv("../backend/model.csv")

params = fine_tuned_model(surprise_input, model = "SVD", cv = 5)

print("Attempting to export the model...")


algo = SVD(**params.best_params['rmse'])
dump.dump("../backend/model/model.pkl", algo=algo)


