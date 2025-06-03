package xyz.ifilk.distributed

import xyz.ifilk.nn.DModule
import xyz.ifilk.optim.Optimizer
import java.io.Serializable

fun interface OptimizerBuilder : Serializable {
    operator fun invoke(params: DModule): Optimizer
}