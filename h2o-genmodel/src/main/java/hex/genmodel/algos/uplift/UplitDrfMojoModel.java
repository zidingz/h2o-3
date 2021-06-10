package hex.genmodel.algos.uplift;

import hex.ModelCategory;
import hex.genmodel.GenModel;
import hex.genmodel.PredictContributions;
import hex.genmodel.algos.tree.SharedTreeMojoModelWithContributions;
import hex.genmodel.algos.tree.TreeSHAPPredictor;


/**
 * "Uplift Distributed Random Forest" MojoModel
 */
public final class UplitDrfMojoModel extends SharedTreeMojoModelWithContributions {
    protected String _uplift_column;
//    protected UpliftMetricType _uplift_metric;
//    public AUUC.AUUCType _auuc_type = AUUC.AUUCType.AUTO;


    public UplitDrfMojoModel(String[] columns, String[][] domains, String responseColumn) {
        super(columns, domains, responseColumn);
    }

    @Override
    protected PredictContributions getContributionsPredictor(TreeSHAPPredictor<double[]> treeSHAPPredictor) {
        return new ContributionsPredictorDRF(this, treeSHAPPredictor);
    }

    /**
     * Corresponds to `hex.tree.drf.DrfMojoModel.score0()`
     */
    @Override
    public final double[] score0(double[] row, double offset, double[] preds) {
        super.scoreAllTrees(row, preds);
        return unifyPreds(row, offset, preds);
    }


    @Override
    public final double[] unifyPreds(double[] row, double offset, double[] preds) {
        return preds;
    }

    @Override
    public double[] score0(double[] row, double[] preds) {
        return score0(row, 0.0, preds);
    }

    static class ContributionsPredictorDRF extends SharedTreeContributionsPredictor {

        private final float _featurePlusBiasRatio;
        private final int _normalizer;

        private ContributionsPredictorDRF(UplitDrfMojoModel model, TreeSHAPPredictor<double[]> treeSHAPPredictor) {
            super(model, treeSHAPPredictor);
            if (ModelCategory.Regression.equals(model._category)) {
                _featurePlusBiasRatio = 0;
                _normalizer = model._ntree_groups;
            } else if (ModelCategory.Binomial.equals(model._category)) {
                _featurePlusBiasRatio = 1f / (model._nfeatures + 1);
                _normalizer = -model._ntree_groups;
            } else
                throw new UnsupportedOperationException(
                        "Model category " + model._category + " cannot be used to calculate feature contributions.");
        }

        @Override
        public float[] getContribs(float[] contribs) {
            for (int i = 0; i < contribs.length; i++) {
                contribs[i] = _featurePlusBiasRatio + (contribs[i] / _normalizer);
            }
            return contribs;
        }
    }

    @Override
    public String[] getOutputNames() {
        return new String[]{"uplift_predict"};
    }

}
