import xyz.ifilk.optim.Adam
import xyz.ifilk.optim.SGD
import xyz.ifilk.tensor.Tensor
import kotlin.test.Test

class OptimTest {
    @Test
    fun testSGDStep() {
        val param = Tensor(doubleArrayOf(1.0, 2.0), intArrayOf(2), requiresGrad = true)
        param.grad = Tensor(doubleArrayOf(0.1, 0.1), intArrayOf(2))

        val optimizer = SGD(listOf(param), lr = 0.5)
        optimizer.step()

        assertArrayEqualsWithTolerance(doubleArrayOf(0.95, 1.95), param.data)
    }

    @Test
    fun testAdamStep() {
        val param = Tensor(doubleArrayOf(1.0, 2.0), intArrayOf(2), requiresGrad = true)
        param.grad = Tensor(doubleArrayOf(0.1, 0.2), intArrayOf(2))

        val optimizer = Adam(listOf(param), lr = 0.001)

        optimizer.step()

        // 由于 Adam 有动量缓存，第一次 step 的变化不会很大，我们只验证下降趋势
        val updated = param.data
        println("Updated params: ${updated.toList()}")

        // 预期值：小于初始值
        assert(updated[0] < 1.0) { "param[0] should decrease" }
        assert(updated[1] < 2.0) { "param[1] should decrease" }

        val param0Before = param.data[0]
        val param1Before = param.data[1]

        // 再次设置梯度，继续下降
        param.grad = Tensor(doubleArrayOf(0.1, 0.2), intArrayOf(2))
        optimizer.step()

        val param0After = param.data[0]
        val param1After = param.data[1]

        assert(param0After < param0Before) { "param[0] should continue to decrease" }
        assert(param1After < param1Before) { "param[1] should continue to decrease" }
    }

}