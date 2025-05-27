
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import xyz.ifilk.nn.DenseLayout
import xyz.ifilk.functions.ReLU
import xyz.ifilk.functions.Sigmoid
import xyz.ifilk.tensor.Tensor
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class LayoutTest {

    @Test
    fun testForwardWithoutBias() {
        val dense = DenseLayout(inFeatures = 3, outFeatures = 2, useBias = false)

        // 手动设置权重，方便验证
        dense.weight.data = doubleArrayOf(
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        )

        val input = Tensor(doubleArrayOf(1.0, 1.0, 1.0), intArrayOf(1, 3), requiresGrad = true)
        val output = dense.forward(input)

        // 输出 = input * weight = [1 1 1] * [[1 2], [3 4], [5 6]] = [1+3+5, 2+4+6] = [9, 12]
        assertEquals(1, output.shape[0])
        assertEquals(2, output.shape[1])
        assertEquals(9.0, output.data[0], 1e-6)
        assertEquals(12.0, output.data[1], 1e-6)
    }

    @Test
    fun testBackwardWithBias() {
        val dense = DenseLayout(inFeatures = 2, outFeatures = 2, useBias = true)

        // 设置权重和偏置
        dense.weight.data = doubleArrayOf(
            1.0, 2.0,
            3.0, 4.0
        )
        dense.bias!!.data = doubleArrayOf(1.0, 1.0)

        val input = Tensor(doubleArrayOf(1.0, 1.0), intArrayOf(1, 2), requiresGrad = true)

        val output = dense.forward(input)  // shape: [1,2]
        val loss = output.sum()            // scalar loss
        loss.backward()

        // 手动推导梯度:
        // dL/dW = input^T * dL/dout = [1,1]^T * [1,1] = [[1,1], [1,1]]
        // dL/db = [1,1]
        // dL/dinput = dL/dout * W^T = [1,1] * [[1,3],[2,4]]^T = [1*1 + 1*2, 1*3 + 1*4] = [3,7]
//        AutogradEngine.printComputationGraph(loss)

        assertNotNull(dense.weight.grad)
        assertNotNull(dense.bias!!.grad)
        assertNotNull(input.grad)

        val expectedGradW = doubleArrayOf(1.0, 1.0, 1.0, 1.0)
        val expectedGradB = doubleArrayOf(1.0, 1.0)
        val expectedGradInput = doubleArrayOf(3.0, 7.0)

        assertArrayEqualsWithTolerance(expectedGradW, dense.weight.grad!!.data)
        assertArrayEqualsWithTolerance(expectedGradB, dense.bias!!.grad!!.data)
        assertArrayEqualsWithTolerance(expectedGradInput, input.grad!!.data)
    }

    @Test
    fun testReLUForwardBackward() {
        val input = Tensor(doubleArrayOf(-1.0, 0.0, 2.0, 3.0), intArrayOf(2, 2), requiresGrad = true)
        val output = ReLU.apply(input)
        val expectedForward = doubleArrayOf(0.0, 0.0, 2.0, 3.0)
        assertArrayEqualsWithTolerance(expectedForward, output.data)

        // 聚合成标量损失
        val loss = output.sum()
        loss.backward()

        // 手动计算梯度，ReLU 梯度是输入 > 0 的地方为 1，否则为 0
        val expectedGrad = doubleArrayOf(0.0, 0.0, 1.0, 1.0)
        assertNotNull(input.grad)
        assertArrayEqualsWithTolerance(expectedGrad, input.grad!!.data)
    }

    @Test
    fun testSigmoidForwardBackward() {
        val input = Tensor(doubleArrayOf(0.0, 2.0, -2.0), intArrayOf(3), requiresGrad = true)
        val output = Sigmoid.apply(input)
        val expectedForward = input.data.map { x -> 1.0 / (1.0 + kotlin.math.exp(-x)) }.toDoubleArray()
        assertArrayEqualsWithTolerance(expectedForward, output.data)

        val loss = output.sum()
        loss.backward()

        val expectedGrad = DoubleArray(output.data.size) { i ->
            val y = output.data[i]
            1.0 * y * (1 - y)  // dLoss/dOutput = 1 since loss = sum(output)
        }

        assertNotNull(input.grad)
        assertArrayEqualsWithTolerance(expectedGrad, input.grad!!.data)
    }

}
