from surprise import dump
import os
from pprint import pprint as pp


def load_model(model_filename):
    print(">> Loading dump")
    from surprise import dump
    import os
    file_name = os.path.expanduser(model_filename)
    _, loaded_model = dump.load(file_name)
    print(">> Loaded dump")
    return loaded_model


def item_rating(model,user, item, verbose = False):


    uid = str(user)
    iid = str(item)
    prediction = model.predict(user, item)
    rating = prediction.est
    details = prediction.details
    uid = prediction.uid
    iid = prediction.iid
    true = prediction.r_ui

    ret = {
        'user': user,
        'item': item,
        'rating': rating,
        'details': details,
        'uid': uid,
        'iid': iid,
        'true': true
    }

    return ret

