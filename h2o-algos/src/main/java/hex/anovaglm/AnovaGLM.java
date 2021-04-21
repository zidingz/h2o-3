package hex.anovaglm;

import hex.ModelBuilder;
import hex.ModelCategory;
import hex.glm.GLMModel;
import water.Key;
import water.exceptions.H2OModelBuilderIllegalArgumentException;

import static hex.glm.GLMModel.GLMParameters.Family.*;

public class AnovaGLM extends ModelBuilder<AnovaGLMModel, AnovaGLMModel.AnovaGLMParameters, AnovaGLMModel.AnovaGLMOutput> {
  private int _nfolds;
  public AnovaGLM(boolean startup_once) { super (new AnovaGLMModel.AnovaGLMParameters(), startup_once); }

  public AnovaGLM(AnovaGLMModel.AnovaGLMParameters parms) {
    super(parms);
    init(false);
  }
  
  public AnovaGLM(AnovaGLMModel.AnovaGLMParameters parms, Key<AnovaGLMModel> key) {
    super(parms, key);
    init(false);
  }
  
  @Override
  protected int nModelsInParallel(int folds) {  // disallow nfold cross-validation
    return nModelsInParallel(1, 2);
  }
  
  @Override
  protected AnovaGLMDriver trainModelImpl() { return new AnovaGLMDriver(); }

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ModelCategory.Regression};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }
  
  @Override
  public boolean haveMojo() { return false; }
  
  @Override
  public boolean havePojo() { return false; }
  
  public BuilderVisibility buildVisibility() { return BuilderVisibility.Experimental; }
  
  public void init(boolean expensive) {
    super.init(expensive);
    _nfolds = _parms._nfolds;
    _parms._nfolds = 0;   // cv is enabled in GLM, not here
    if (expensive)
      validateAnovaGLMParameters();
  }
  
  private void validateAnovaGLMParameters() {
    if (gaussian != _parms._family && tweedie != _parms._family && AUTO != _parms._family)
      error("_family", " only gaussian and tweedie families are supported here");
    
    if (_parms._link == null)
      _parms._link = GLMModel.GLMParameters.Link.family_default;
    
    if (error_count() > 0)
      throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(AnovaGLM.this);
  }
  
  private class AnovaGLMDriver extends Driver {
    Key<AnovaGLMModel>[] glmModels; // store GLM models built

    @Override
    public void computeImpl() {
      init(true);
    }
  }
}
