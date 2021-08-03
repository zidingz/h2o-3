setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

infogramIris <- function() {
    browser()
    bhexFV <- h2o.importFile(locate("smalldata/admissibleml_test/irisROriginal.csv"))
    Y <- "Species"
    X <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
    deepCMI <- c(0.1038524, 0.7135458, 0.5745915, 1.0000000)
    deepRel <- c(0.009010006, 0.011170417, 0.755170945, 1.000000000)
    Log.info("Build the model")
    mFV <- h2o.infogram(y = Y, x = X, training_frame = bhexFV, distribution = "multinomial", seed = 12345)

    Log.info("Check that the columns used in the model are the ones we passed in.")

    Log.info("===================Columns passed in: ================")
    Log.info(paste("index ", X, " ", names(bhexFV)[X], "\n", sep = ""))
    Log.info("======================================================")
    preds <- mFV@model$coefficients_table$names
    preds <- preds[2:length(preds)]
    Log.info("===================Columns Used in Model: =========================")
    Log.info(paste(preds, "\n", sep = ""))
    Log.info("================================================================")

    expect_that(preds, equals(colnames(bhexFV)[X]))

}

doTest("Infogram: Iris core infogram", infogramIris)

