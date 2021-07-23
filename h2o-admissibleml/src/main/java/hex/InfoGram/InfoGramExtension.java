package hex.InfoGram;

import water.AbstractH2OExtension;

public class InfoGramExtension extends AbstractH2OExtension {

  @Override
  public String getExtensionName() {
    return "InfoGram";
  }

  @Override
  public void init() {
    new InfoGram(true); // as a side effect DummyModelBuilder will be registered in a static field ModelBuilder.ALGOBASES
  }
}
