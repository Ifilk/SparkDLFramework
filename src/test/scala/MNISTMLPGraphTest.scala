
import org.junit.jupiter.api.Test
import xyz.ifilk.autograd.ToolsKt.toDotGraph
import xyz.ifilk.models.mnist.MNISTMLP
import xyz.ifilk.utils.TensorUtilKt.rand

import java.io.{BufferedWriter, FileWriter}
import scala.util.Using

class MNISTMLPGraphTest {

  @Test
  def MNISTMLPGraph(): Unit = {
    // 构建模型
    val model = new MNISTMLP()

    // 构造伪输入（batchSize = 1, 28x28 图像）
    val input = rand(1, 1, 28, 28)
    input.setRequiresGrad(true) // 确保有计算图追踪

    // 前向传播
    val output = model.forward(input).sum()

    // 触发反向图（否则部分节点不会出现在 dot 图中）
    output.backward(null)

    // 获取 dot 格式字符串
    val dotGraph = toDotGraph(output)

    // 写入到文件
    Using.resource(new BufferedWriter(new FileWriter("visual.dot"))) { writer =>
      writer.write(dotGraph)
    }

    println("✅ Computation graph written to visual.html")
  }
}
