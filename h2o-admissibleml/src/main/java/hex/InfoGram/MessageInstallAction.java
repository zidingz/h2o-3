package hex.InfoGram;

import water.DKV;
import water.Key;
import water.parser.BufferedString;

public class MessageInstallAction extends InfoGramAction<MessageInstallAction> {
  private final Key _trgt;
  private final String _msg;

  public MessageInstallAction(Key trgt, String msg) {
    _trgt = trgt;
    _msg = msg;
  }

  @Override
  protected String run(InfoGramModel.InfoGramParameter parms) {
    DKV.put(_trgt, new BufferedString("Computed " + _msg));
    return _msg;
  }

  @Override
  protected void cleanUp() {
    DKV.remove(_trgt);
  }
}
