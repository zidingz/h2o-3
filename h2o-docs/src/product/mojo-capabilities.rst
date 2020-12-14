.. _mojo-capabilities:

MOJO Capabilities
-----------------

This section describes basics of every day work with MOJO model in H2O-3.

Gen-model package breakdown?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source `here <productionizing.html>`__ provides a lot of information about MOJOs and how to get them.
In this place is a breakdown the package documentation `POJO and MOJO Model Javadoc <http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/index.html>`__.
The h2o-genmodel API contains:

- **hex.genmodel.algos**

All algorithms supports MOJO can be found here. These models can be load and directly used to scoring or other model
specific actions. For the documentation of the methods refer to javadoc of `GenModel class <http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/hex/genmodel/GenModel.html>`__.

- **hex.genmodel.easy.EasyPredictModelWrapper**

Wrapper for MOJO models to get easier and readable interface for scoring and other model specific actions.

- **hex.genmodel.easy.prediction**

All types of predictions that can EasyPredictModelWrapper.predict(). For more information refer to the `javadoc of the classes <http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/hex/genmodel/easy/prediction/AbstractPrediction.html>`__.

- **hex.genmodel.easy.CategoricalEncoder**

Classes from this interface can be used for preprocess raw data values to proper Categorical values what model expects.

- **hex.genmodel.attributes.metrics**

Different metrics for model specific needs.

- **hex.genmodel.tools**

Java command line tools for various type of application. For example tools for printing decision trees, reading a CSV file and making predictions, reading a CSV file and munging it, and more..


How to predict value with mojo?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::
   .. code-tab:: java Raw MOJO

        MojoModel mojoModel = MojoModel.load("isolation_forest.zip");
        double [] predictions = new double[]{Double.NaN, Double.NaN};
        mojoModel.score0(new double[]{100, 100}, predictions);
        System.out.println(Arrays.toString(predictions));

   .. code-tab:: java Mojo Wrapper

        IsolationForestMojoModel mojoModel = (IsolationForestMojoModel) MojoModel.load("isolation_forest.zip");

        EasyPredictModelWrapper.Config config = new EasyPredictModelWrapper.Config()
                .setModel(mojoModel);
        EasyPredictModelWrapper model = new EasyPredictModelWrapper(config);

        RowData row = new RowData();
        row.put("x", "100");
        row.put("y", "100");

        AnomalyDetectionPrediction p = (AnomalyDetectionPrediction) model.predict(row);
        System.out.println("[" + p.normalizedScore + ", " + p.score + "]");



What metadata MOJO model contain?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All of h2o-3 models contains some metadata. You can print them in python by calling:

.. tabs::
    .. code-tab:: r R

        Just write model variable name to console.

    .. code-tab:: python

        # All metadata
        model._model_json

        # Useful metadata
        model._model_json['output']

        # Description of available keys
        model._model_json['output']['help'

    .. code-tab:: Java Java MOJO Model

        // Must call with metadata flag set to True
        MojoModel model = MojoModel.load("GBM_model.zip", true);
        ModelAttributes attributes = model._modelAttributes;

All of the MOJO models contain `following attributes <http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/hex/genmodel/attributes/ModelAttributes.html>`__:

    .. code:: java

        // Correspond to model._model_json['output']['model_summary'] (Number of trees, Size of model,..)
        attributes.getModelSummary();

        // Correspond to model._model_json['output']['scoring_history']
        attributes.getScoringHistory();

        // Correspond to model._model_json['output']['training_metrics']
        // but only some values are available (MSE, RMSE,...)
        // and for example confusion Matrix and other is omitted.
        attributes.getTrainingMetrics();

        // Correspond to model._model_json['output']['validation_metrics']
        // but only some values are available (MSE, RMSE,...)
        // and for example confusion Matrix and other is omitted.
        attributes.getValidationMetrics();

        // Correspond to model._model_json['output']['cross_validation_metrics']
        // but only some values are available (MSE, RMSE,...)
        // and for example confusion Matrix and other is omitted.
        attributes.getCrossValidationMetrics();

        // Correspond to model._model_json['output']['cross_validation_metrics_summary']
        attributes.getCrossValidationMetricsSummary();

        // Model parameters setting when the model was built
        attributes.getModelParameters();


In example bellow is a way how to get number of trees from model.


.. tabs::
   .. code-tab:: r R

      model <- h2o.randomForest(...)
      print(paste("Number of Trees: ", model@allparameters$ntrees))

   .. code-tab:: python

      model = H2ORandomForestEstimator(...)
      model.train(...)
      print("Number of Trees: {}".format(model._model_json["output"]["model_summary"]["number_of_trees"]))

   .. code-tab:: Java Java MOJO Model

      MojoModel model = MojoModel.load("rf_model.zip", true);
      ModelAttributes attributes = model._modelAttributes;
      System.out.print(attributes.getModelSummary().getColHeaders()[1] + ": ");
      System.out.println(attributes.getModelSummary().getCell(1,0));


Subclasses of ModelAttributes are used to handle model specific attributes, for example variable importance:


.. tabs::
   .. code-tab:: java Raw MOJO

        // Must call with metadata flag set to True
        MojoModel model = MojoModel.load("GBM_model.zip", true);
        SharedTreeModelAttributes attributes = ((SharedTreeModelAttributes) model._modelAttributes);
        String[] variables = attributes.getVariableImportances()._variables;
        double[] importances = attributes.getVariableImportances()._importances;
        System.out.print(variables[0] + ": ");
        System.out.println(importances[0]);

   .. code-tab:: java Mojo Wrapper

        MojoModel modelMojo = MojoModel.load("GBM_model.zip", true);
        EasyPredictModelWrapper.Config config = new EasyPredictModelWrapper.Config().setModel(modelMojo);
        EasyPredictModelWrapper model = new EasyPredictModelWrapper(config);
        KeyValue[] importances = model.varimp();
        System.out.print(importances[0].getKey() + ": ");
        System.out.println(importances[0].getValue());









