import matplotlib.pyplot as plt
import numpy as np
import time
import h2o
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator, H2OUpliftRandomForestEstimator
from causalml.dataset import make_uplift_classification
from tests import pyunit_utils

seed = 1234
ntrees = 100
attempt_per_thread = 3  # number of runs of the algorithm in the thread
threds = [12, 10, 8, 6, 4, 2, 1]
nbins = 1000
nbins_top_level = 1024
max_depth = 16


def syntetic_data(n, p):
    train, x_names = make_uplift_classification(n_samples=n,
                                                treatment_name=['control', 'treatment'],
                                                n_classification_features=p,
                                                n_classification_informative=p,
                                                random_seed=seed
                                                )

    treatment_column = "treatment_group_key"
    response_column = "conversion"

    return train, x_names, treatment_column, response_column


def train_models(nthreads, data, treatment_column, response_column, x_names,  max_depth, start_cluster=True):
    if start_cluster:
        h2o.init(nthreads=nthreads)
    start = time.time()
    hf = h2o.H2OFrame(data)
    hf[treatment_column] = hf[treatment_column].asfactor()
    hf[response_column] = hf[response_column].asfactor()
    uplift_h2o = H2OUpliftRandomForestEstimator(
        ntrees=ntrees,
        max_depth=max_depth,
        treatment_column=treatment_column,
        uplift_metric="KL",
        distribution="bernoulli",
        gainslift_bins=10,
        min_rows=10,
        nbins_top_level=nbins_top_level,
        nbins=nbins,
        seed=seed,
        sample_rate=0.99,
        auuc_type="gain"
    )
    uplift_h2o.train(y=response_column, x=x_names, training_frame=hf)
    end = time.time()
    uplift_time = end - start
    print(f"Uplift Time: {uplift_time}s")
    start = time.time()
    rf_h2o = H2ORandomForestEstimator(
        ntrees=ntrees,
        max_depth=max_depth,
        distribution="bernoulli",
        gainslift_bins=10,
        min_rows=10,
        nbins_top_level=nbins_top_level,
        nbins=nbins,
        seed=seed,
        sample_rate=0.99,
        binomial_double_trees=True
    )
    # rf_h2o.train(y=response_column, x=x_names, training_frame=hf)
    end = time.time()
    rf_time = end - start
    print(f"RF Time: {rf_time}s")
    start = time.time()
    gbm = H2OGradientBoostingEstimator(
        ntrees=ntrees,
        max_depth=max_depth,
        distribution="bernoulli",
        gainslift_bins=10,
        min_rows=10,
        nbins_top_level=nbins_top_level,
        nbins=nbins,
        seed=seed,
        sample_rate=0.99,
    )
    gbm.train(y=response_column, x=x_names, training_frame=hf)
    end = time.time()
    gbm_time = end - start
    print(f"gbm Time: {gbm_time}s")
    if start_cluster:
        h2o.cluster().shutdown()
    return uplift_time, rf_time, gbm_time


def run_benchmark(n, p, start_cluster=True):
    data, x_names, treatment_column, response_column = syntetic_data(n, p)
    all_times = []
    all_times_num_uplift = []
    all_times_num_rf = []
    all_times_num_gbm = []
    for nthreads in threds:
        times_uplift = []
        times_rf = []
        times_gbm = []
        for i in range(attempt_per_thread):
            uplift_time, rf_time, gbm_time = train_models(nthreads, data, treatment_column, response_column, x_names,  max_depth, start_cluster)
            times_uplift.append(uplift_time)
            times_rf.append(rf_time)
            times_gbm.append(gbm_time)
        print(f"Uplift {np.mean(times_uplift)}s")
        print(f"RF {np.mean(times_rf)}s")
        print(f"gbm {np.mean(times_gbm)}s")
        all_times_num_uplift.append(times_uplift)
        all_times_num_rf.append(times_rf)
        all_times_num_gbm.append(times_gbm)
        all_times.append(f"thread {nthreads} - Uplift {np.mean(times_uplift)}s and RF {np.mean(times_rf)}s and gbm {np.mean(times_gbm)}s")

    uplift_means = dict()
    rf_means = dict()
    gbm_means = dict()
    for i, nthreads in enumerate(threds):
        print(f"{nthreads} - Uplift = {np.mean(all_times_num_uplift[i])}, RF = {np.mean(all_times_num_rf[i])}, gbm = {np.mean(all_times_num_gbm[i])}")
        uplift_means[nthreads] = np.mean(all_times_num_uplift[i])
        rf_means[nthreads] = np.mean(all_times_num_rf[i])
        gbm_means[nthreads] = np.mean(all_times_num_gbm[i])

    return uplift_means, rf_means, gbm_means


def plot_result(n, p, uplift_means, rf_means, gbm_means):
    data = {"x":[], "y": [], "label":[]}
    for label, coord in uplift_means.items():
        data["x"].append(label)
        data["y"].append(coord)

    data_if = {"x":[], "y": [], "label":[]}
    for label, coord in rf_means.items():
        data_if["x"].append(label)
        data_if["y"].append(coord)

    data_gbm = {"x":[], "y": [], "label":[]}
    for label, coord in gbm_means.items():
        data_gbm["x"].append(label)
        data_gbm["y"].append(coord)

    fig=plt.figure(figsize=(8, 10))
    fig.add_subplot(111)
    plt.plot(data['x'], data['y'], '-', label="UpliftRF", linewidth=3)
    plt.plot(data_if['x'], data_if['y'], '-', label="RF", linewidth=3)
    plt.plot(data_gbm['x'], data_gbm['y'], '-', label="GBM", linewidth=3)
    plt.xlabel("Number of threads")
    plt.ylabel("Computing time (s)")
    plt.legend()
    plt.tick_params(direction='out', length=6, width=2)
    plt.title(f"Uplift Random Forest - training benchmark\nModel: n = {n}; p = {p}; ntrees = {ntrees};  max_depth = {max_depth}; nbins = {nbins}")
    plt.savefig(f"h2o-scale-perf_{max_depth}_{n}_{coord}.png", bbox_inches='tight', pad_inches=.05)
    plt.show()


def uplift_bench():
    n = 1000
    p = 5
    uplift_time, rf_time, gbm_time = run_benchmark(n, p, False)
    plot_result(n, p, uplift_time, rf_time, gbm_time)


if __name__ == "__main__":
    pyunit_utils.standalone_test(uplift_bench)
else:
    uplift_bench()
