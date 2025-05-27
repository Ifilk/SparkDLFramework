package xyz.ifilk.autograd

import xyz.ifilk.tensor.Tensor
import java.util.*

class AutogradEngine {
    companion object  {
        /** Reverse‑mode autodiff using DFS topological ordering. */
        fun backward(tensor: Tensor) {
            // 1. Build a root‑first topological ordering using DFS with visited set
            val topo = ArrayList<Tensor>()
            val visited = HashSet<Tensor>()
            val stack = Stack<Tensor>()
            stack.push(tensor)
            while (stack.isNotEmpty()) {
                val node = stack.pop()
                if (!visited.add(node)) continue
                topo.add(node)
                if (node.creator != null)
                    for (input in node.creator!!.inputs)
                        stack.push(input)
            }
            // 2. Propagate gradients along the topological order (root → leaves)
            for (node in topo) {
                if (node.creator == null || node.grad == null) continue
                val grads = node.creator!!.backward(node.grad!!)
                for (i in grads.indices) {
                    val inp = node.creator!!.inputs[i]
                    if (inp.grad != null) {
                        // in-place operation
                        inp.grad!!.add_(grads[i])
                    } else inp.grad = grads[i]
                }
            }
        }

        /**
         * Prints the computation graph starting from the given tensor.
         * @param tensor The root tensor of the computation graph
         * @param indent Current indentation level (used for recursion)
         */
        fun printComputationGraph(tensor: Tensor, indent: Int = 0) {
            val indentStr = "  ".repeat(indent)

            println("${indentStr}Tensor(shape=${tensor.shape.contentToString()}, grad=${tensor.grad != null})")

            tensor.creator?.let { creator ->
                println("${indentStr}↑ created by: ${creator::class.simpleName}")
                creator.inputs.forEach { input ->
                    printComputationGraph(input, indent + 1)
                }
            }
        }

        /**
         * Generates a string representation of the computation graph in DOT format.
         * This can be visualized using Graphviz.
         * @param tensor The root tensor of the computation graph
         * @return DOT format string representing the computation graph
         */
        fun toDotGraph(tensor: Tensor): String {
            val nodes = mutableSetOf<String>()
            val edges = mutableSetOf<String>()
            val stack = Stack<Tensor>()
            val visited = HashSet<Tensor>()

            stack.push(tensor)
            visited.add(tensor)

            while (stack.isNotEmpty()) {
                val current = stack.pop()
                val nodeId = "node${System.identityHashCode(current)}"

                nodes.add("""
                    $nodeId [label=<
                        <table border="0" cellborder="1" cellspacing="0">
                            <tr><td colspan="2"><b>${current.shape.contentToString()}</b></td></tr>
                            <tr><td colspan="2">${if (current.grad != null) "Grad" else "No Grad"}</td></tr>
                            ${if (current.creator != null) "<tr><td colspan=\"2\">${current.creator!!::class.simpleName}</td></tr>" else ""}
                        </table>
                    >];
                """.trimIndent())

                current.creator?.inputs?.forEach { input ->
                    val inputId = "node${System.identityHashCode(input)}"
                    edges.add("$inputId -> $nodeId")

                    if (!visited.contains(input)) {
                        visited.add(input)
                        stack.push(input)
                    }
                }
            }

            return """
                digraph computation_graph {
                    rankdir="BT";
                    node [shape=plaintext];
                    ${nodes.joinToString("\n")}
                    ${edges.joinToString("\n")}
                }
            """.trimIndent()
        }
    }
}