import org.junit.jupiter.api.Assertions.assertEquals
import kotlin.math.abs
import kotlin.test.assertTrue

fun assertArrayEqualsWithTolerance(expected: DoubleArray, actual: DoubleArray, tol: Double = 1e-6) {
    assertEquals(expected.size, actual.size)
    for (i in expected.indices) {
        assertTrue(abs(expected[i] - actual[i]) < tol, "Mismatch at $i: expected=${expected[i]}, actual=${actual[i]}")
    }
}