#############################################
## H2O Random Forest vs. uplift.upliftRF test
#############################################

setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")


test.uplift <- function() {
    library(uplift)
    ### simulate data for uplift modeling

    set.seed(123)
    train <- sim_pte(n = 1000, p = 6, rho = 0, sigma = sqrt(2), beta.den = 4)
    train$treat <- ifelse(train$treat == 1, 1, 0)
    
    ntrees <- 1000
    
    print("Train data summary")
    print(summary(train))

    print("Uplift fit model")
    # fit upliftRF
    modelUplift <- upliftRF(y ~ X1 + X2 + X3 + X4 + X5 + X6 + trt(treat),
        data = train,
        split_method = "KL",
        mtry = 6,
        ntree = ntrees,
        #interaction.depth = 10,
        minsplit = 10,
        min_bucket_ct0 = 10,
        min_bucket_ct1 = 10,
        verbose = TRUE)
    
    print("Uplift model summary")
    print(summary(modelUplift))

    # predict upliftRF on new data
    print("Uplift predict on test data")
    test <- sim_pte(n = 200, p = 20, rho = 0, sigma =  sqrt(2), beta.den = 4)
    test$treat <- ifelse(test$treat == 1, 1, 0)
    predUplift <- predict(modelUplift, train)
    print(head(predUplift))

    # fit h2o RF
    train$treat <- as.factor(train$treat)
    train$y <- as.factor(train$y)
    trainH2o <- as.h2o(train)
    modelH2o <- h2o.randomForest(x = c("X1", "X2", "X3", "X4", "X5", "X6"), y = "y",
        training_frame = trainH2o,
        uplift_column = "treat",
        uplift_metric = "KL",
        distribution = "bernoulli",
        gainslift_bins = 10,
        ntrees = ntrees,
        max_depth = 10,
        min_rows = 10,
        nbins = 100,
        seed = 42)

    print(h2o.gainsLift(modelH2o))
    print(h2o.varimp(modelH2o))
    
    # predict upliftRF on new data for treatment group
    testH2oTreat <- as.h2o(train)
    predH2oTreat <- predict(modelH2o, testH2oTreat)
    print(head(predH2oTreat))

    res <- as.data.frame(predH2oTreat)
    res$pr.y1_ct1 <- predUplift[,1]
    res$pr.y1_ct0 <- predUplift[,2]
    res$div_ct1 <- res$p0 - res$pr.y1_ct1
    res$div_ct0 <- res$p1 - res$pr.y1_ct0
    print(head(res))
    print(summary(res))

    
}

doTest("Random Forest Test: Test H2O RF uplift against uplift.upliftRF", test.uplift)
