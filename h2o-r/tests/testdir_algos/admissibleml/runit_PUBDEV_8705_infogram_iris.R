setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

# tests that infogram build the correct model for core infogram.  Make sure 
# 1. it gets the correct result compared to deep's code.
# 2. the relevance and cmi frame contains the correct values
infogramIris <- function() {
    browser()
    bhexFV <- h2o.importFile(locate("smalldata/admissibleml_test/irisROriginal.csv"))
    Y <- "Species"
    X <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
    deepCMI <- sort(c(0.1038524, 0.7135458, 0.5745915, 1.0000000))
    deepRel <- sort(c(0.009010006, 0.011170417, 0.755170945, 1.000000000))
    Log.info("Build the model")
    mFV <- h2o.infogram(y = Y, x = X, training_frame = bhexFV, distribution = "multinomial", seed = 12345)
    relCMIFrame <- h2o.get_relevance_cmi_frame(mFV) # get frames containing relevance and cmi
    allCMI <- h2o.get_all_predictor_cmi(mFV)
    allRel <- h2o.get_all_predictor_relevance(mFV)
    admissibleCMI <- h2o.get_admissible_cmi(mFV)
    admissibleRel <- get_admissible_relevance(mFV)
    
    expect_equal(deepCMI, sor(allCMI), tolerance=1e-6)
    expect_equal(deepRel, sort(allRel), tolerance=1e-6)
    expect_true(admissibleCMI >= 0.1)
    expect_true(admissibleRel >= 0.1)
}

doTest("Infogram: Iris core infogram", infogramIris)
