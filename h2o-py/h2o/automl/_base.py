import h2o
from h2o.model import ModelBase
from h2o.utils.typechecks import assert_is_type


class H2OAutoMLBaseMixin:
    def predict(self, test_data):
        """
        Predict on a dataset.

        :param H2OFrame test_data: Data on which to make predictions.

        :returns: A new H2OFrame of predictions.

        :examples:
        >>> # Set up an H2OAutoML object
        >>> aml = H2OAutoML(max_runtime_secs=30)
        >>> # Launch an H2OAutoML run
        >>> aml.train(y=y, training_frame=train)
        >>> # Predict with top model from AutoML Leaderboard on a H2OFrame called 'test'
        >>> aml.predict(test)
        >>>
        >>> # Get AutoML object by `project_name`
        >>> get_aml = h2o.automl.get_automl(aml.project_name)
        >>> # Predict with top model from AutoML Leaderboard on a H2OFrame called 'test'
        >>> get_aml.predict(test)
        """
        return self.leader.predict(test_data)

    # ---------------------------------------------------------------------------
    # Download POJO/MOJO with AutoML
    # ---------------------------------------------------------------------------
    def download_pojo(self, path="", get_genmodel_jar=False, genmodel_name=""):
        """
        Download the POJO for the leader model in AutoML to the directory specified by path.

        If path is an empty string, then dump the output to screen.

        :param path:  An absolute path to the directory where POJO should be saved.
        :param get_genmodel_jar: if True, then also download h2o-genmodel.jar and store it in folder ``path``.
        :param genmodel_name: Custom name of genmodel jar
        :returns: name of the POJO file written.
        """
        return h2o.download_pojo(self.leader, path, get_jar=get_genmodel_jar, jar_name=genmodel_name)

    def download_mojo(self, path=".", get_genmodel_jar=False, genmodel_name=""):
        """
        Download the leader model in AutoML in MOJO format.

        :param path: the path where MOJO file should be saved.
        :param get_genmodel_jar: if True, then also download h2o-genmodel.jar and store it in folder ``path``.
        :param genmodel_name: Custom name of genmodel jar
        :returns: name of the MOJO file written.
        """
        return ModelBase.download_mojo(self.leader, path, get_genmodel_jar, genmodel_name)

    @property
    def project_name(self):
        """
        Retrieve a string indicating the project_name of the automl instance to retrieve.

        :return: a string containing the project_name
        """
        pass

    @property
    def leader(self):
        """
        Retrieve the top model from an H2OAutoML object

        :return: an H2O model

        :examples:
        >>> # Set up an H2OAutoML object
        >>> aml = H2OAutoML(max_runtime_secs=30)
        >>> # Launch an AutoML run
        >>> aml.train(y=y, training_frame=train)
        >>> # Get the best model in the AutoML Leaderboard
        >>> aml.leader
        >>>
        >>> # Get AutoML object by `project_name`
        >>> get_aml = h2o.automl.get_automl(aml.project_name)
        >>> # Get the best model in the AutoML Leaderboard
        >>> get_aml.leader
        """
        pass

    @property
    def leaderboard(self):
        """
        Retrieve the leaderboard from an H2OAutoML object

        :return: an H2OFrame with model ids in the first column and evaluation metric in the second column sorted
                 by the evaluation metric

        :examples:
        >>> # Set up an H2OAutoML object
        >>> aml = H2OAutoML(max_runtime_secs=30)
        >>> # Launch an AutoML run
        >>> aml.train(y=y, training_frame=train)
        >>> # Get the AutoML Leaderboard
        >>> aml.leaderboard
        >>>
        >>> # Get AutoML object by `project_name`
        >>> get_aml = h2o.automl.get_automl(aml.project_name)
        >>> # Get the AutoML Leaderboard
        >>> get_aml.leaderboard
        """
        pass

    @property
    def training_info(self):
        """
        Expose the name/value columns of `event_log` as a simple dictionary, for example `start_epoch`, `stop_epoch`, ...
        See :func:`event_log` to obtain a description of those key/value pairs.

        :return: a dictionary with event_log['name'] column as keys and event_log['value'] column as values.
        """
        pass

    @property
    def event_log(self):
        """
        Retrieve the backend event log from an H2OAutoML object

        :return: an H2OFrame with detailed events occurred during the AutoML training.
        """
        pass

    def get_best_model(self, algorithm=None, criterion=None):
        """
        Get best model of a given family/algorithm for a given criterion from an AutoML object.

        :param algorithm: One of "basemodel", "deeplearning", "drf", "gbm", "glm", "stackedensemble", "xgboost".
                          If None, pick the best model regardless of the algorithm.
        :param criterion: Criterion can be one of the metrics reported in leaderboard. If set to None, the same ordering
                          as in the leaderboard will be used.
                          Avaliable criteria:
                            - Regression metrics: deviance, rmse, mse, mae, rmsle
                            - Binomial metrics: auc, logloss, aucpr, mean_per_class_error, rmse, mse
                            - Multinomial metrics: mean_per_class_error, logloss, rmse, mse
                          The following additional leaderboard information can be also used as a criterion:
                            - 'training_time_ms': column providing the training time of each model in milliseconds (doesn't include the training of cross validation models).
                            - 'predict_time_per_row_ms`: column providing the average prediction time by the model for a single row.
        :return: An H2OModel or None if no model of a given family is present
        :examples:
        >>> # Set up an H2OAutoML object
        >>> aml = H2OAutoML(max_runtime_secs=30)
        >>> # Launch an AutoML run
        >>> aml.train(y=y, training_frame=train)
        >>> gbm = aml.get_best_model("gbm")
        """
        from h2o.exceptions import H2OValueError
        def _get_models(leaderboard):
            return [m[0] for m in
                    leaderboard["model_id"].as_data_frame(use_pandas=False, header=False)]

        higher_is_better = ["auc", "aucpr"]

        assert_is_type(algorithm, None, str)
        assert_is_type(criterion, None, str)

        if criterion is not None:
            criterion = criterion.lower()

        if "deviance" == criterion:
            criterion = "mean_residual_deviance"

        if algorithm is not None:
            if algorithm.lower() not in ("basemodel", "deeplearning", "drf", "gbm",
                                         "glm", "stackedensemble", "xgboost"):
                raise H2OValueError("Algorithm \"{}\" is not supported!".format(algorithm))
            algorithm = algorithm.lower()

        extra_cols = ["algo"]
        if criterion in ("training_time_ms", "predict_time_per_row_ms"):
            extra_cols.append(criterion)

        leaderboard = h2o.automl.get_leaderboard(self, extra_columns=extra_cols)
        leaderboard = leaderboard if algorithm is None else (
            leaderboard[leaderboard["algo"].tolower() == algorithm, :] if algorithm != "basemodel"
            else leaderboard[leaderboard["algo"].tolower() != "stackedensemble", :])

        if leaderboard.nrow == 0:
            return None

        if criterion is None:
            return h2o.get_model(leaderboard[0, "model_id"])

        if criterion not in leaderboard.columns:
            raise H2OValueError("Criterion \"{}\" is not present in the leaderboard!".format(criterion))

        models_in_default_order = _get_models(leaderboard)
        sorted_lb = leaderboard.sort(by=criterion, ascending=criterion not in higher_is_better)
        selected_models = _get_models(sorted_lb[sorted_lb[criterion] == sorted_lb[0, criterion]])
        picked_model = [model for model in models_in_default_order if model in selected_models][0]

        return h2o.get_model(picked_model)
