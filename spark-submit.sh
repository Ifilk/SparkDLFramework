#!/usr/bin/env bash
set -euo pipefail

SPARK_MASTER_URL=${SPARK_MASTER_URL:-spark://spark-master:7077}
HBASE_HOST=${HBASE_HOST:-hbase}
HBASE_PORT=${HBASE_PORT:-16010}
MNIST_TABLE=${MNIST_TABLE:-mnist_dataset}
APP_JAR=${APP_JAR:-/opt/app/app.jar}
MAIN_CLASS=${MAIN_CLASS:-xyz.ifilk.dataset.mnist.MNISTTrain}
ZK_HOST=${ZK_HOST:-hbase}
HBASE_ZOOKEEPER_QUORUM=${HBASE_ZOOKEEPER_QUORUM:-hbase}

echo "Waiting for Spark Master at $SPARK_MASTER_URL ..."
until curl -s http://spark-master:8080 > /dev/null; do
  echo "Spark Master not available yet. Retrying in 3s..."
  sleep 3
done
echo "Spark Master is up"

echo "Waiting for HBase at $HBASE_HOST:$HBASE_PORT ..."
until nc -z "$HBASE_HOST" "$HBASE_PORT"; do
  echo "HBase not available yet. Retrying in 3s..."
  sleep 3
done
echo "HBase is up"

echo "Waiting for 'init' container to become healthy..."
while [ ! -f /opt/app/hbase-init/init_done.flag ]; do
  echo "[submit] Waiting for init to complete..."
  sleep 3
done
echo "[submit] Init completed"

# 提交 Spark 作业
echo "Submitting Spark job..."
/opt/bitnami/spark/bin/spark-submit \
  --class "$MAIN_CLASS" \
  --master "$SPARK_MASTER_URL" \
  --deploy-mode client \
  --total-executor-cores 4 \
  --executor-memory 2G \
  --conf spark.executorEnv.HBASE_ZOOKEEPER_QUORUM="$HBASE_ZOOKEEPER_QUORUM" \
  "$APP_JAR"

echo "Spark job submitted successfully."
