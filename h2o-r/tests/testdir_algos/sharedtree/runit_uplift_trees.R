#############################################
## H2O Random Forest vs. uplift.upliftRF test
#############################################

setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")


test.uplift <- function() {
    library(uplift)
    ### simulate data for uplift modeling

    set.seed(123)
    train <- sim_pte(n = 100, p = 6, rho = 0, sigma = sqrt(2), beta.den = 4)
    train$treat <- ifelse(train$treat == 1, 1, 0)
    
    print("Train data summary")
    print(summary(train))

    print("Uplift fit model")
    # fit upliftRF
    modelUplift <- upliftRF(y ~ X1 + X2 + X3 + X4 + X5 + X6 + trt(treat),
        data = train,
        split_method = "KL",
        mtry = 6,
        ntree = 50,
        interaction.depth = 20,
        minsplit = 20,
        verbose = TRUE)
    
    print("Uplift model summary")
    print(summary(modelUplift))

    # predict upliftRF on new data
    print("Uplift predict on test data")
    test <- sim_pte(n = 200, p = 20, rho = 0, sigma =  sqrt(2), beta.den = 4)
    test$treat <- ifelse(test$treat == 1, 1, 0)

    predUplift <- predict(modelUplift, test)
    
    print("Uplift prediction head")
    print(head(predUplift))

    print("Uplift performance")
    perf <- performance(predUplift[, 1], predUplift[, 2], test$y, test$treat, direction = 1)
    print(perf)
    #plot(perf[, 8] ~ perf[, 1], type ="l", xlab = "Decile", ylab = "uplift")

    # fit h2o RF
    trainH2o <- as.h2o(train)
    modelH2o <- h2o.randomForest(x = c("X1", "X2", "X3", "X4", "X5", "X6"), y = "y",
        training_frame = trainH2o,
        uplift_column = "treat",
        uplift_metric = "KL",
        ntrees = 50,
        max_depth = 20,
        min_rows = 20)

    # predict upliftRF on new data
    testH2o <- as.h2o(test)
    predH2o <- predict(modelH2o, testH2o)
    print(head(predH2o))
}

doTest("Random Forest Test: Test H2O RF uplift against uplift.upliftRF", test.uplift)
