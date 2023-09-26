import os
import pandas as pd
from surprise import BaselineOnly
from surprise import NormalPredictor
from surprise import SVD
from surprise import SVDpp
from surprise.model_selection import cross_validate
from tqdm import tqdm


def model_search(data, selected_model=None, cv=2, metrics='mae'):
    """
    Search for the best algorithm among the given models using cross-validation.

    :param data: The dataset for modeling.
    :param selected_model: A list of selected models to evaluate, or None to use default models.
    :param cv: Number of cross-validation folds.
    :param metrics: Evaluation metric to use (e.g., 'mae').
    :return: A sorted DataFrame containing algorithm evaluation results.
    """

    benchmark = []

    # Iterate over all algorithms
    if selected_model is None:
        algorithms = [
            SVDpp(), SVD(), BaselineOnly(), NormalPredictor()
        ]
    else:
        algorithms = selected_model

    print("Attempting: ", str(algorithms), '\n\n\n')

    # Perform modelling on every selected_model or algorithms
    for algorithm in tqdm(algorithms):
        print("Starting: ", str(algorithm))

        # Perform cross validation
        results = cross_validate(algorithm, data, measures=[metrics.upper()], cv=cv, verbose=False)

        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)

        tmp = pd.concat([
            pd.Series([
                str(algorithm).split(' ')[0].split('.')[-1],
            ]), tmp,
        ],
        )
        benchmark.append(tmp)
        print("Done: ", str(algorithm), "\n\n")

    print('\n\tDONE\n')

    # Converting benchmark to pandas dataframe
    results = pd.DataFrame(benchmark)
    results['weighted_sum'] = (results[f'test_{metrics}'] * 100) + (0.5 * results.iloc[:, 1:].sum(axis=1))
    return results.sort_values(
        by=['weighted_sum'],
        ascending=[True]
    ).iloc[:,:-1]


def model_train(train=True, model_data='results', file_name=None):
    """
    Train and return a selected model based on the saved model data.

    :param train: Boolean flag indicating whether to training a new model or load an existing one.
    :param model_data: Path to the directory containing model data.
    :param file_name: Name of the file containing the model information.
    :return: Trained model object.
    """

    model_dict = {

        'BaselineOnly': BaselineOnly(),
        'NormalPredictor': NormalPredictor(),
        'SVD': SVD(),
        'SVDpp': SVDpp(),

    }

    data_path = os.path.join(
        os.path.dirname(__file__),
        model_data,
        file_name
    )

    result = pd.read_csv(data_path)
    model_used = result.iloc[0, 0]
    model = model_dict[model_used]

    return model


def hybrid_recsys(data):
    pass


if __name__ == "__main__":
    print(model_train())