set -euo pipefail

SPARK_MASTER_URL=${SPARK_MASTER_URL:-spark://spark-master:7077}
HBASE_HOST=${HBASE_HOST:-hbase}
HBASE_PORT=${HBASE_PORT:-16010}
ZK_HOST=${ZK_HOST:-hbase}
MNIST_TABLE=${MNIST_TABLE:-mnist_dataset}

echo "⏳ Waiting for Spark Master at $SPARK_MASTER_URL ..."
until curl -s http://spark-master:8080 > /dev/null; do
  echo "⏳ Spark Master not available yet. Retrying in 3s..."
  sleep 3
done
echo "✅ Spark Master is up"

echo "⏳ Waiting for HBase at $HBASE_HOST:$HBASE_PORT ..."
until nc -z "$HBASE_HOST" "$HBASE_PORT"; do
  echo "⏳ HBase not available yet. Retrying in 3s..."
  sleep 3
done
echo "✅ HBase is up"

echo "📦 Importing MNIST to HBase (if not already exists)..."
python3 /opt/app/load_mnist.py \
  --zk "$ZK_HOST" \
  --images /opt/app/train-images-idx3-ubyte.gz \
  --labels /opt/app/train-labels-idx1-ubyte.gz \
  --table "$MNIST_TABLE" || echo "⚠️ MNIST table may already exist. Continuing..."
