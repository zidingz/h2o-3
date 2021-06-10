package hex.genmodel.algos.uplift;

import hex.genmodel.algos.tree.SharedTreeMojoReader;

import java.io.IOException;

/**
 */
public class UpliftDrfMojoReader extends SharedTreeMojoReader<UplitDrfMojoModel> {

  @Override
  public String getModelName() {
    return "Distributed Random Forest";
  }

  @Override
  protected void readModelData() throws IOException {
    super.readModelData();
    _model._uplift_column = readkv("uplift_column");
  }

  @Override
  protected UplitDrfMojoModel makeModel(String[] columns, String[][] domains, String responseColumn) {
    return new UplitDrfMojoModel(columns, domains, responseColumn);
  }

  @Override public String mojoVersion() {
    return "1.40";
  }
}
