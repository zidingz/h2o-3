package hex.InfoGram;

import water.Iced;

public abstract class InfoGramAction<T> extends Iced<InfoGramAction<T>> {
  protected abstract String run(InfoGramModel.InfoGramParameter parms);
  protected void cleanUp() {};
}
