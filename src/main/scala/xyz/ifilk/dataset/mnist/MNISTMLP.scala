package xyz.ifilk.dataset.mnist

import xyz.ifilk.nn.{DModule, LinearLayout}
import xyz.ifilk.tensor.Tensor

import java.util

class MNISTMLP extends DModule {

  val fc1 = new LinearLayout(784, 128, true)
  val fc2 = new LinearLayout(128, 10, true)

  // 初始化添加参数
  fc1.getParameters.forEach(n => registerParameter(n))
  fc2.getParameters.forEach(n => registerParameter(n))

  override def forward(input: Tensor): Tensor = {
    val batchSize = input.getShape()(0)
    val flat = input.reshape(batchSize, 784) // 扁平化为 [batchSize, 784]
    var x = fc1.forward(flat)
    x = x.relu()
    x = fc2.forward(x)
    x.softmax(-1)
  }

  override def loadParameters(value: Array[Tensor]): Unit = {
    val fc1Params = fc1.getParameters.size()
    val fc2Params = fc2.getParameters.size()
// java.lang.NoSuchMethodError
//    fc1.loadParameters(value.slice(0, fc1Params))
//    fc2.loadParameters(value.slice(fc1Params, fc1Params + fc2Params))
    fc1.loadParameters(util.Arrays.copyOfRange(value, 0, fc1Params))
    fc2.loadParameters(util.Arrays.copyOfRange(value, fc1Params, fc1Params + fc2Params))
  }
}
