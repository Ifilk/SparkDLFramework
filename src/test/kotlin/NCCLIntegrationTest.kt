import org.junit.jupiter.api.*
import xyz.ifilk.distributed.nccl.api.NCCLComm
import xyz.ifilk.distributed.nccl.server.NCCLServer
import xyz.ifilk.tensor.Tensor
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class NCCLIntegrationTest {

    private val worldSize = 4
    private val port = 9999
    private lateinit var serverThread: Thread

    @BeforeAll
    fun startServer() {
        serverThread = Thread {
            NCCLServer(port).start()
        }.apply { isDaemon = true; start() }
        Thread.sleep(500)          // 等待服务端就绪
    }

    @AfterAll
    fun stopServer() {
        serverThread.interrupt()   // 仅用于示例; 生产环境请给 NCCLServer 添加优雅关闭接口
    }

    /* ---------------- helper ---------------- */
    private fun tensorOf(vararg v: Double) = Tensor(v, intArrayOf(v.size))

    /* ---------------- Tests ----------------- */

    @Test
    fun barrierBroadcastAllReduceGatherReduceScatter() {
        val pool = Executors.newFixedThreadPool(worldSize)
        val resultsBroadcast = Array<Tensor?>(worldSize) { null }
        val resultsAllReduce = Array<Tensor?>(worldSize) { null }
        val resultsGather    = Array<Tensor?>(worldSize) { null }
        val resultsRS        = Array<Tensor?>(worldSize) { null }

        repeat(worldSize) { rank ->
            pool.submit {
                val comm = NCCLComm.init("jobX", worldSize, rank)

                /* barrier --------------------------------------------------- */
                comm.barrier()   // 所有线程到齐即可继续

                /* broadcast ------------------------------------------------- */
                val bcastSrc = if (rank == 0) tensorOf(42.0, 43.0) else tensorOf(0.0, 0.0)
                val bcastOut = comm.broadcast(bcastSrc, root = 0)
                resultsBroadcast[rank] = bcastOut

                /* allReduce(sum) ------------------------------------------- */
                val arInput = tensorOf(rank + 1.0, rank + 1.0)  // 1..4
                val arOut   = comm.allReduce(arInput)           // sum = [10,10]
                resultsAllReduce[rank] = arOut

                /* gather (root=0) ------------------------------------------ */
                val gOut = comm.gather(tensorOf(rank.toDouble()), root = 0)
                resultsGather[rank] = gOut     // root0 得到 concat，其余得到空 Tensor

                /* reduceScatter -------------------------------------------- */
                val rsInput = tensorOf(1.0, 1.0, 1.0, 1.0)      // len==worldSize
                val rsOut   = comm.reduceScatter(rsInput)       // each rank gets [4/worldSize]=1 element
                resultsRS[rank] = rsOut
            }
        }

        pool.shutdown()
        pool.awaitTermination(10, TimeUnit.SECONDS)

        /* ---------- Assertions ---------- */
        /* broadcast */
        repeat(worldSize) { rank ->
            assertContentEquals(doubleArrayOf(42.0, 43.0), resultsBroadcast[rank]!!.data,
                "Broadcast mismatch at rank $rank")
        }

        /* allReduce */
        repeat(worldSize) { rank ->
            assertContentEquals(doubleArrayOf(10.0, 10.0), resultsAllReduce[rank]!!.data,
                "AllReduce mismatch at rank $rank")
        }

        /* gather: rank0 concat [0,1,2,3], others empty (size0) */
        assertContentEquals(doubleArrayOf(0.0,1.0,2.0,3.0), resultsGather[0]!!.data)
        (1 until worldSize).forEach { r ->
            assertEquals(0, resultsGather[r]!!.data.size, "Gather non-root should be empty")
        }

        /* reduceScatter: input all ones, sum=4, each rank gets 1  */
        repeat(worldSize) { rank ->
            assertContentEquals(doubleArrayOf(4.0), resultsRS[rank]!!.data,
                "ReduceScatter mismatch rank $rank")
        }
    }
}
