from __future__ import print_function
import os
import sys

from h2o.estimators.infogram import H2OInfogram

sys.path.insert(1, os.path.join("..","..",".."))
import h2o
from tests import pyunit_utils
    
def test_infogram_breast_cancer_cv_valid():
    """
    Test to make sure validation frame and cross-validation are implemented properly.
    """
    fr = h2o.import_file(path=pyunit_utils.locate("smalldata/admissibleml_test/wdbc_changed.csv"))
    target = "diagnosis"
    fr[target] = fr[target].asfactor()
    
    x = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
         "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
         "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
         "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
         "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
         "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]
    splits = fr.split_frame(ratios=[0.80])
    train = splits[0]
    test = splits[1]
    infogram_model_v = H2OInfogram(seed = 12345, top_n_features=50) # model with validation dataset
    infogram_model_v.train(x=x, y=target, training_frame=train, validation_frame=test)
    relcmi_train_v = infogram_model_v.get_relevance_cmi_frame()
    relcmi_valid_v = infogram_model_v.get_relevance_cmi_frame(valid=True)
    
    infogram_model = H2OInfogram(seed = 12345, top_n_features=50) # model with no validation and cross-validation
    infogram_model.train(x=x, y=target, training_frame=train)
    relcmi_train = infogram_model.get_relevance_cmi_frame()
    
    infogram_model_cv = H2OInfogram(seed = 12345, top_n_features=50, nfolds=3) # model with cross-validation
    infogram_model_cv.train(x=x, y=target, training_frame=train)
    relcmi_train_cv = infogram_model_cv.get_relevance_cmi_frame()
    relcmi_cv_cv = infogram_model_cv.get_relevance_cmi_frame(cv=True)
    
    infogram_model_cv_valid = H2OInfogram(seed = 12345, top_n_features=50, nfolds=3)
    infogram_model_cv_valid.train(x=x, y=target, training_frame=train, validation_frame=test)
    relcmi_train_cv_valid = infogram_model_cv_valid.get_relevance_cmi_frame()
    relcmi_valid_cv_valid = infogram_model_cv_valid.get_relevance_cmi_frame(valid=True)
    relcmi_cv_cv_valid = infogram_model_cv_valid.get_relevance_cmi_frame(cv=True)
    
    # training rel cmi frames should all equal
    print("Comparing infogram data from training dataset")
    pyunit_utils.compare_frames_local(relcmi_train_v, relcmi_train, prob=1)
    pyunit_utils.compare_frames_local(relcmi_train_cv, relcmi_train_cv_valid, prob=1)
    pyunit_utils.compare_frames_local(relcmi_train_v, relcmi_train_cv, prob=1)
    
    # cv rel cmi frames should be the same
    print("Comparing infogram data from cross-validation dataset")
    pyunit_utils.compare_frames_local(relcmi_cv_cv, relcmi_cv_cv_valid, prob=1)
    
    # valid frames should equal
    print("Comparing infogram data from validation dataset")
    pyunit_utils.compare_frames_local(relcmi_valid_cv_valid, relcmi_valid_v, prob=1)

if __name__ == "__main__":
    pyunit_utils.standalone_test(test_infogram_breast_cancer_cv_valid)
else:
    test_infogram_breast_cancer_cv_valid()
