from __future__ import print_function
import sys, os

sys.path.insert(1, os.path.join("..", "..", ".."))
import h2o
from tests import pyunit_utils
from h2o.estimators import H2OUpliftRandomForestEstimator
from causalml.dataset import make_uplift_classification


def uplift_random_forest_save_and_load():
    print("Uplift Distributed Random Forest Save Load Test")

    # Data generation
    # Generate dataset with 10 features, treatment/control flag feature
    # and outcome feature (In this case "conversion")
    train, x_names = make_uplift_classification(n_samples=500,
                                                treatment_name=['control', 'treatment'],
                                                n_classification_features=10,
                                                # Dataset contains only features valid for modeling
                                                # Do not confuse model with irrelevant or redundant features
                                                n_classification_informative=10,
                                                random_seed=12345
                                                )

    treatment_column = "treatment_group_key"
    response_column = "conversion"

    train_h2o = h2o.H2OFrame(train)
    train_h2o[treatment_column] = train_h2o[treatment_column].asfactor()
    train_h2o[response_column] = train_h2o[response_column].asfactor()

    uplift_model = H2OUpliftRandomForestEstimator(
        ntrees=10,
        max_depth=5,
        uplift_column=treatment_column,
        uplift_metric="KL",
        distribution="bernoulli",
        gainslift_bins=10,
        min_rows=10,
        nbins=1000,
        seed=42,
        sample_rate=0.99,
        auuc_type="gain"
    )
    uplift_model.train(y=response_column, x=x_names, training_frame=train_h2o)
    prediction = uplift_model.predict(train_h2o)
    print(prediction)
    uplift_predict = prediction['uplift_predict'].as_data_frame(use_pandas=True)["uplift_predict"]

    path = pyunit_utils.locate("results")

    assert os.path.isdir(path), "Expected save directory {0} to exist, but it does not.".format(path)
    model_path = h2o.save_model(uplift_model, path=path, force=True)

    assert os.path.isfile(model_path), "Expected load file {0} to exist, but it does not.".format(model_path)
    reloaded = h2o.load_model(model_path)
    prediction_reloaded = reloaded.predict(train_h2o)
    print(prediction_reloaded)
    uplift_predict_reloaded = prediction_reloaded['uplift_predict'].as_data_frame(use_pandas=True)["uplift_predict"]

    assert isinstance(reloaded,
                      H2OUpliftRandomForestEstimator), \
        "Expected and H2OUpliftRandomForestEstimator, but got {0}" \
        .format(reloaded)

    assert uplift_predict[0] == uplift_predict_reloaded[0], "Output is not the same after reload"
    assert uplift_predict[5] == uplift_predict_reloaded[5], "Output is not the same after reload"
    assert uplift_predict[33] == uplift_predict_reloaded[33], "Output is not the same after reload"
    assert uplift_predict[256] == uplift_predict_reloaded[256], "Output is not the same after reload"
    assert uplift_predict[499] == uplift_predict_reloaded[499], "Output is not the same after reload"


if __name__ == "__main__":
    pyunit_utils.standalone_test(uplift_random_forest_save_and_load)
else:
    uplift_random_forest_save_and_load()
