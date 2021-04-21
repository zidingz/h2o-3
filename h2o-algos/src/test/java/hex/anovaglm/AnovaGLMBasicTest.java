package hex.anovaglm;

import hex.Model;
import hex.glm.GLMModel;
import org.junit.Test;
import org.junit.runner.RunWith;
import water.DKV;
import water.Scope;
import water.fvec.Frame;
import water.runner.CloudSize;
import water.runner.H2ORunner;

import static hex.glm.GLMModel.GLMParameters.Family.gaussian;
import static water.TestUtil.parseTestFile;

@RunWith(H2ORunner.class)
@CloudSize(1)
public class AnovaGLMBasicTest {

  // first test to make sure CV can be properly specified
  @Test
  public void testAnovaGLM1() {
    try {
      Scope.enter();
      Frame train = parseTestFile("smalldata/iris/iris_train.csv");
      DKV.put(train);
      Scope.track(train);

      AnovaGLMModel.AnovaGLMParameters params = new AnovaGLMModel.AnovaGLMParameters();
      params._family = gaussian;
      params._response_column = "sepal_len";
      params._train = train._key;
      params._solver = GLMModel.GLMParameters.Solver.IRLSM;
      params._fold_assignment = Model.Parameters.FoldAssignmentScheme.Random;
      params._nfolds = 3;

      AnovaGLMModel anovaG = new AnovaGLM(params).trainModel().get();
      Scope.track_generic(anovaG);
    } finally {
      Scope.exit();
    }
  }
}
