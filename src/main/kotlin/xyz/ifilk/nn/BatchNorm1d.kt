package xyz.ifilk.nn

import xyz.ifilk.tensor.Tensor
import xyz.ifilk.utils.copyFrom
import kotlin.math.sqrt

/**
 * 对形状为 [batch, numFeatures] 的 2-D 张量做批归一化。
 *
 * @param numFeatures 每个样本的特征维度
 * @param momentum    估计运行均值/方差时的动量
 * @param eps         为了数值稳定在方差上加的常数
 * @param affine      是否学习 scale (γ) 与 shift (β) 参数
 */
class BatchNorm1d(
    private val numFeatures: Int,
    private val momentum: Double = 0.1,
    private val eps: Double = 1e-5,
    private val affine: Boolean = true,
) : DModule() {

    /* 可训练参数 ------------------------------------------------------ */
    /** γ，初始为 1 */
    val weight = if (affine) Parameter(
        name = "bn.weight",
        data = DoubleArray(numFeatures) { 1.0 },
        shape = intArrayOf(numFeatures),
        requiresGrad = true
    ) else null

    /** β，初始为 0 */
    val bias = if (affine) Parameter(
        name = "bn.bias",
        data = DoubleArray(numFeatures) { 0.0 },
        shape = intArrayOf(numFeatures),
        requiresGrad = true
    ) else null

    /* 运行时统计量（Buffer，不参与梯度） ------------------------------- */
    private var runningMean = Tensor(DoubleArray(numFeatures) { 0.0 }, intArrayOf(numFeatures))
    private var runningVar  = Tensor(DoubleArray(numFeatures) { 1.0 }, intArrayOf(numFeatures))

    init {
        if (affine) {
            registerParameter(weight!!)
            registerParameter(bias!!)
        }
        // 假设 DModule 已提供 registerBuffer；如没有可去掉
        registerBuffer("running_mean", runningMean)
        registerBuffer("running_var", runningVar)
    }

    /* 前向传播 -------------------------------------------------------- */
    override fun forward(input: Tensor): Tensor {
        // input: [N, F]
        require(input.shape.size == 2 && input.shape[1] == numFeatures) {
            "BatchNorm1d expects input shape [batch, $numFeatures], got ${input.shape.toList()}"
        }

        val (mean, varUnbiased) = if (isTraining) {
            /* step①: 计算 batch 内均值与方差 */
            val m = input.mean(axis = 0)               // [F]
            val centered = input - m                   // [N, F]
            val varB = (centered * centered).mean(axis = 0) // [F]

            /* step②: 更新 Running 统计量 */
            runningMean *= Tensor(1 - momentum, requiresGrad = false)
            runningMean += m * momentum
            runningVar  *= Tensor(1 - momentum, requiresGrad = false)
            runningVar  += varB * momentum

            m to varB
        } else {
            runningMean to runningVar
        }

        /* step③: 归一化 */
        val invStd = (varUnbiased + eps).sqrt().reciprocal() // [F]
        var output = (input - mean) * invStd                 // [N, F]

        /* step④: 仿射变换（可选） */
        if (affine) {
            output = output * weight!! + bias!!
        }
        return output
    }

    /* 加载参数 -------------------------------------------------------- */
    override fun loadParameters(value: Array<Tensor>) {
        if (!affine) return      // 无可训练参数时直接返回
        require(value.size == 2) { "Expected 2 tensors (gamma, beta) but got ${value.size}" }
        weight!!.copyFrom(value[0])
        bias!!.copyFrom(value[1])
    }

    /* 可选：重置 Running 统计量 ---------------------------------------- */
    fun resetRunningStats() {
        runningMean.fill(0.0)
        runningVar.fill(1.0)
    }
}
