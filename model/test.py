from model.data_interface.data import *
from model.inference.training import *


data = load_data(data_path="../data/dicoding_user_item_rating.gzip")
print(data.head())


data_input = data.loc[:,["user_id", "course_id", "rating"]]
surprise_input = reader_data(data=data_input,
                             cols=["user_id", "course_id", "rating"],
                             scale=True,
                             model = "surprise")


print((type(surprise_input)))
model = model_search(data = surprise_input, cv = 3)
model.to_csv("../backend/model.csv")

print(model)


