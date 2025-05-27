package xyz.ifilk.utils

import xyz.ifilk.tensor.Tensor

fun haveSameShape(t1: Tensor, t2: Tensor): Boolean {
    // Handle scalar cases
    val shape1 = normalizeShape(t1.shape)
    val shape2 = normalizeShape(t2.shape)

    // Compare normalized shapes
    return shape1.contentEquals(shape2)
}

// Helper function to normalize shapes (e.g., [] and [1] both represent scalars)
private fun normalizeShape(shape: IntArray): IntArray {
    return when {
        // Empty shape (scalar) becomes [1]
        shape.isEmpty() -> intArrayOf(1)
        // Shape with single 1 (e.g., [1], [1,1]) gets reduced
        shape.all { it == 1 } -> IntArray(1) { 1 }
        else -> shape
    }
}