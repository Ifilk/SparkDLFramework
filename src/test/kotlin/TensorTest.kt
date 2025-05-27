import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import xyz.ifilk.tensor.Tensor

class TensorTest {
    @Test
    fun testAdditionForward() {
        val a = Tensor(doubleArrayOf(1.0, 2.0), intArrayOf(2), false)
        val b = Tensor(doubleArrayOf(3.0, 4.0), intArrayOf(2), false)
        val c = a + b
        assertArrayEquals(doubleArrayOf(4.0, 6.0), c.data, 1e-6)
    }

    @Test
    fun testMultiplicationForward() {
        val a = Tensor(doubleArrayOf(1.0, 2.0), intArrayOf(2), false)
        val b = Tensor(doubleArrayOf(3.0, 4.0), intArrayOf(2), false)
        val c = a * b
        assertArrayEquals(doubleArrayOf(3.0, 8.0), c.data, 1e-6)
    }

    @Test
    fun testMatMulForward() {
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0), intArrayOf(3), false)
        val b = Tensor(doubleArrayOf(4.0, 5.0, 6.0), intArrayOf(3), false)
        val c = a mm b
        assertArrayEquals(doubleArrayOf(32.0), c.data, 1e-6)
    }

    @Test
    fun testSimpleBackwardAdd() {
        val a = Tensor(doubleArrayOf(2.0), intArrayOf(1), true)
        val b = Tensor(doubleArrayOf(3.0), intArrayOf(1), true)
        val c = a + b
        c.backward()
        assertEquals(1.0, a.grad?.data!![0], 1e-6)
        assertEquals(1.0, b.grad?.data!![0], 1e-6)
    }

    @Test
    fun testSimpleBackwardMul() {
        val a = Tensor(doubleArrayOf(2.0), intArrayOf(1), true)
        val b = Tensor(doubleArrayOf(3.0), intArrayOf(1), true)
        val c = a * b
        c.backward()
        assertEquals(3.0, a.grad?.data!![0], 1e-6)
        assertEquals(2.0, b.grad?.data!![0], 1e-6)
    }

    @Test
    fun testChainRule() {
        // z = (a + b) * c
        val a = Tensor(doubleArrayOf(2.0), intArrayOf(1), true)
        val b = Tensor(doubleArrayOf(3.0), intArrayOf(1), true)
        val c = Tensor(doubleArrayOf(4.0), intArrayOf(1), true)
        val z = (a + b) * c
        z.backward()

        // ∂z/∂a = c = 4, ∂z/∂b = c = 4, ∂z/∂c = a + b = 5
        assertEquals(4.0, a.grad?.data!![0], 1e-6)
        assertEquals(4.0, b.grad?.data!![0], 1e-6)
        assertEquals(5.0, c.grad?.data!![0], 1e-6)
    }

    @Test
    fun testMatmulBackward2x2() {
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2), requiresGrad = true)
        val b = Tensor(doubleArrayOf(5.0, 6.0, 7.0, 8.0), intArrayOf(2, 2), requiresGrad = true)
        val c = a mm b  // shape: [2,2]
        val loss = c.sum()   // 简单起见，将输出所有元素求和作为标量损失
        loss.backward()

        // 手动计算梯度：
        // dL/dA = dC/dA = 1 * B^T
        val expectedGradA = doubleArrayOf(
            11.0, 15.0,
            11.0, 15.0
        ) // shape: [2,2]

        // dL/dB = dC/dB = 1 * A^T
        val expectedGradB = doubleArrayOf(
            4.0, 4.0,
            6.0, 6.0
        ) // shape: [2,2]

        assertNotNull(a.grad)
        assertNotNull(b.grad)
        assertArrayEqualsWithTolerance(expectedGradA, a.grad!!.data)
        assertArrayEqualsWithTolerance(expectedGradB, b.grad!!.data)
    }
}