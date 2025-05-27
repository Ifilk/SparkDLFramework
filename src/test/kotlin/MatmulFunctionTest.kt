import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertThrows
import xyz.ifilk.tensor.Tensor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MatmulFunctionTest {

    @Test
    fun testMatrixMatrixMultiplication() {
        // (2x3) · (3x2) → (2x2)
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), intArrayOf(2, 3))
        val b = Tensor(doubleArrayOf(7.0, 8.0, 9.0, 10.0, 11.0, 12.0), intArrayOf(3, 2))

        val result = a mm b

        assertEquals(2, result.shape[0])
        assertEquals(2, result.shape[1])
        assertEquals(58.0, result.data[0])  // 1*7 + 2*9 + 3*11
        assertEquals(64.0, result.data[1])  // 1*8 + 2*10 + 3*12
        assertEquals(139.0, result.data[2]) // 4*7 + 5*9 + 6*11
        assertEquals(154.0, result.data[3]) // 4*8 + 5*10 + 6*12
    }

    @Test
    fun testMatrixVectorMultiplication() {
        // (2x3) · (3) → (2)
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), intArrayOf(2, 3))
        val b = Tensor(doubleArrayOf(7.0, 8.0, 9.0), intArrayOf(3))

        val result = a mm b

        assertEquals(1, result.shape.size)
        assertEquals(2, result.shape[0])
        assertEquals(50.0, result.data[0])  // 1*7 + 2*8 + 3*9
        assertEquals(122.0, result.data[1]) // 4*7 + 5*8 + 6*9
    }

    @Test
    fun testVectorMatrixMultiplication() {
        // (3) · (3x2) → (2)
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0), intArrayOf(3))
        val b = Tensor(doubleArrayOf(7.0, 8.0, 9.0, 10.0, 11.0, 12.0), intArrayOf(3, 2))

        val result = MatmulFunction.apply(a, b)

        assertEquals(1, result.shape.size)
        assertEquals(2, result.shape[0])
        assertEquals(58.0, result.data[0])  // 1*7 + 2*9 + 3*11
        assertEquals(64.0, result.data[1])  // 1*8 + 2*10 + 3*12
    }

    @Test
    fun testBackwardWithSumReduction() {
        // For non-scalar outputs, we need to reduce to scalar first to call backward()
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2), requiresGrad = true)
        val b = Tensor(doubleArrayOf(5.0, 6.0, 7.0, 8.0), intArrayOf(2, 2), requiresGrad = true)

        val result = a mm b
        // Reduce to scalar by summing all elements
        val sumResult = result.sum()
        sumResult.backward()

        // dL/dA (sum of gradients)
        assertTrue(a.grad != null)
        assertArrayEquals(doubleArrayOf(11.0, 15.0, 11.0, 15.0), a.grad!!.data, 1e-6)

        // dL/dB
        assertTrue(b.grad != null)
        assertArrayEquals(doubleArrayOf(4.0, 4.0, 6.0, 6.0), b.grad!!.data, 1e-6)
    }

    @Test
    fun testNoGrad() {
        val a = Tensor(doubleArrayOf(1.0, 2.0), intArrayOf(2))
        val b = Tensor(doubleArrayOf(3.0, 4.0), intArrayOf(2))

        val result = MatmulFunction.apply(a, b)
        assertTrue(!result.requiresGrad)
    }

    @Test
    fun testIncompatibleShapes() {
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0), intArrayOf(3))
        val b = Tensor(doubleArrayOf(4.0, 5.0), intArrayOf(2))

        assertThrows(IllegalArgumentException::class.java) {
            MatmulFunction.apply(a, b)
        }
    }
}