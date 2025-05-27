import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import xyz.ifilk.tensor.Tensor

class TensorTest {
    @Test
    fun testAdditionForward() {
        val a = Tensor(doubleArrayOf(1.0, 2.0), intArrayOf(2), false)
        val b = Tensor(doubleArrayOf(3.0, 4.0), intArrayOf(2), false)
        val c = a.add(b)
        assertArrayEquals(doubleArrayOf(4.0, 6.0), c.data, 1e-6)
    }

    @Test
    fun testMultiplicationForward() {
        val a = Tensor(doubleArrayOf(1.0, 2.0), intArrayOf(2), false)
        val b = Tensor(doubleArrayOf(3.0, 4.0), intArrayOf(2), false)
        val c = a.mul(b)
        assertArrayEquals(doubleArrayOf(3.0, 8.0), c.data, 1e-6)
    }

    @Test
    fun testMatMulForward() {
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0), intArrayOf(3), false)
        val b = Tensor(doubleArrayOf(4.0, 5.0, 6.0), intArrayOf(3), false)
        val c = a.matmul(b)
        assertArrayEquals(doubleArrayOf(32.0), c.data, 1e-6)
    }

    @Test
    fun testSimpleBackwardAdd() {
        val a = Tensor(doubleArrayOf(2.0), intArrayOf(1), true)
        val b = Tensor(doubleArrayOf(3.0), intArrayOf(1), true)
        val c = a.add(b)
        c.backward()
        assertEquals(1.0, a.grad?.data!![0], 1e-6)
        assertEquals(1.0, b.grad?.data!![0], 1e-6)
    }

    @Test
    fun testSimpleBackwardMul() {
        val a = Tensor(doubleArrayOf(2.0), intArrayOf(1), true)
        val b = Tensor(doubleArrayOf(3.0), intArrayOf(1), true)
        val c = a.mul(b)
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
        val z = a.add(b).mul(c)
        z.backward()

        // ∂z/∂a = c = 4, ∂z/∂b = c = 4, ∂z/∂c = a + b = 5
        assertEquals(4.0, a.grad?.data!![0], 1e-6)
        assertEquals(4.0, b.grad?.data!![0], 1e-6)
        assertEquals(5.0, c.grad?.data!![0], 1e-6)
    }
}