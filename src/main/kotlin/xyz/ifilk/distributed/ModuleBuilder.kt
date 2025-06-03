package xyz.ifilk.distributed

import xyz.ifilk.nn.DModule
import java.io.Serializable

fun interface ModuleBuilder : Serializable {
    operator fun invoke(): DModule
}