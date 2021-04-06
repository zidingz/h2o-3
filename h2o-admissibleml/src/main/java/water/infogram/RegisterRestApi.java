package water.infogram;

import hex.InfoGram.InfoGram;
import water.api.AlgoAbstractRegister;
import water.api.RestApiContext;
import water.api.SchemaServer;

public class RegisterRestApi extends AlgoAbstractRegister {
  @Override
  public void registerEndPoints(RestApiContext context) {
    InfoGram infogramMB = new InfoGram(true);
    // Register InfoGram model builder REST API
    registerModelBuilder(context, infogramMB, SchemaServer.getStableVersion());
  }

  @Override
  public String getName() {
    return "InfoGram";
  }
}
