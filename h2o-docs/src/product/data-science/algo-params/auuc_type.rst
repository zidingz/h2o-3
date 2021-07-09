``auuc_type``
-----
- Available in: Uplift DRF
- Hyperparameter: no


Description
~~~~~~~~~~~

Use this option to specify calculation of uplift and AUUC.

To calculating AUUC aggregated statistics from binned prediction are used: 

1. how many observations are in the treatment group (how many data rows in the bin have ``treatment_column`` label == 1)
2. how many observations are in the control group (how many data rows int he bin have ``treatment_column`` label == 0)
3. how many observations are in the treatment group and response to the offer (how many data rows in the bin have ``treatment_column`` label == 1 and ``response_column`` label == 1)
4. how many observations are in the control group and response to the offer (how many data rows in the bin have ``treatment_column`` label == 0 and ``response_column`` label == 1)


In H2O Uplift DRF three ``auuc_type`` are supported:
- Qini (``auuc_type="qini"``)
- Lift (``auuc_type="lift"``)
- Gain (``auuc_type="gain"``)

Related Parameters
~~~~~~~~~~~~~~~~~~

- `treatment_column <treatment_column.html>`__
- `uplift_metric <uplift_metric.html>`__


Example
~~~~~~~

.. tabs::
   .. code-tab:: r R

    library(h2o)
    h2o.init()

    # Import the uplift dataset into H2O:
    data <- h2o.importFile(locate("https://s3.amazonaws.com/h2o-public-test-data/smalldata/uplift/criteo_uplift_13k.csv"))

    # Set the predictors, response and treatment column
    # set the predictors
    predictors <- c("f1", "f2", "f3", "f4", "f5", "f6","f7", "f8") 
    # set the response as a factor
    data$conversion <- as.factor(data$conversion)
    # set the treatment column as a factor
    data$treatment <- as.factor(data$treatment)

    # Split the dataset into a train and valid set:
    data_split <- h2o.splitFrame(data = data, ratios = 0.8, seed = 1234)
    train <- data_split[[1]]
    valid <- data_split[[2]]

    # Build and train the model:
    uplift.model <- h2o.upliftRandomForest(training_frame = train,
                                           validation_frame=valid,               
                                           x=predictors,
                                           y="conversion",
                                           ntrees=10,
                                           max_depth=5,
                                           treatment_column="treatment",
                                           uplift_metric="KL",
                                           gainslift_bins=10,
                                           min_rows=10,
                                           nbins=1000,
                                           seed=1234,
                                           auuc_type="qini")
    # Eval performance:
    perf <- h2o.performance(uplift.model)

    # Generate predictions on a validation set (if necessary):
    predict <- h2o.predict(uplift.model, newdata = valid)

   .. code-tab:: python
   
    import h2o
    from h2o.estimators import H2OUpliftRandomForestEstimator
    h2o.init()

    # Import the cars dataset into H2O:
    data = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/uplift/criteo_uplift_13k.csv")

    # Set the predictors, response and treatment column
    predictors = ["f1", "f2", "f3", "f4", "f5", "f6","f7", "f8"]
    # set the response as a factor
    response = "conversion"
    data[response] = data[response].asfactor()
    # set the treatment as a factor
    treatment_column = "treatment"
    data[treatment_column] = data[treatment_column].asfactor()

    # Split the dataset into a train and valid set:
    train, valid = data.split_frame(ratios=[.8], seed=1234)

    # Build and train the model:
    uplift_model = H2OUpliftRandomForestEstimator(ntrees=10,
                                                  max_depth=5,
                                                  treatment_column=treatment_column,
                                                  uplift_metric="KL",
                                                  gainslift_bins=10,
                                                  min_rows=10,
                                                  nbins=1000,
                                                  seed=1234,
                                                  auuc_type="gain")
    uplift_model.train(x=predictors, 
                       y=response, 
                       training_frame=train, 
                       validation_frame=valid)

    # Eval performance:
    perf = uplift_model.model_performance()

    # Generate predictions on a validation set (if necessary):
    pred = uplift_model.predict(valid)
