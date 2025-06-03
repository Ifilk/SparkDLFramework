#!/usr/bin/env python3
import gzip
import struct
import argparse
import happybase
import time
import os
import struct

def serialize_tensor(data, shape, requires_grad=False):
    """
    将 Tensor（列表）序列化为与 Kotlin serializeTensor 完全兼容的 ByteArray（Little Endian）
    """
    shape_len = len(shape)
    total_bytes = 4 + shape_len * 4 + 1 + len(data) * 8
    buf = bytearray()

    # shape length
    buf += struct.pack('<i', shape_len)
    for dim in shape:
        buf += struct.pack('<i', dim)

    # requiresGrad
    buf += struct.pack('<?', requires_grad)

    # data: 转 double
    for val in data:
        buf += struct.pack('<d', float(val))

    return bytes(buf)


def read_idx(filename):
    """读取 IDX 格式的 MNIST 文件"""
    with gzip.open(filename, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic == 2051:  # 图像
            rows, cols = struct.unpack(">II", f.read(8))
            data = f.read()
            return ('images', num_items, rows, cols, data)
        elif magic == 2049:  # 标签
            data = f.read()
            return ('labels', num_items, data)
        else:
            raise ValueError(f"Unknown magic number {magic} in file {filename}")

def ensure_table(connection, table_name, column_family='data'):
    tables = connection.tables()
    if table_name.encode() not in tables:
        print(f"Table '{table_name}' does not exist. Creating...")
        connection.create_table(
            table_name,
            {column_family: dict()}
        )
        print(f"Table '{table_name}' created.")
        # 创建后稍微等待一下，避免马上写入失败
        time.sleep(2)
    else:
        print(f"Table '{table_name}' already exists.")

def put_to_hbase(connection, table_name, images, labels):
    table = connection.table(table_name)
    b = table.batch(batch_size=1000)
    for i in range(images[1]):
        row_key = f"train_{i:05d}"
        # 原始 uint8 图像数据
        raw = images[4][i * images[2] * images[3]:(i + 1) * images[2] * images[3]]
        # 转换为 double 数组
        pixel_array = list(raw)  # uint8 to int
        double_array = [float(p) / 255.0 for p in pixel_array]  # 归一化到 [0,1] 也是常见做法
        # uint8 to 4 bytes
        label_byte = struct.pack(">i", labels[2][i])
        img_bytes = serialize_tensor(double_array, shape=[images[2], images[3]], requires_grad=False)

        b.put(row_key.encode(), {
            b"data:image": img_bytes,
            b"data:label": label_byte
        })
    b.send()
    print(f"{images[1]} records written to HBase table '{table_name}'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zk", default="localhost", help="HBase Zookeeper host")
    parser.add_argument("--table", default="mnist_dataset", help="HBase table name")
    parser.add_argument("--images", default="train-images-idx3-ubyte.gz")
    parser.add_argument("--labels", default="train-labels-idx1-ubyte.gz")
    parser.add_argument("--port", type=int, default=9090, help="HBase Thrift port")
    args = parser.parse_args()

    assert os.path.exists(args.images), f"{args.images} not found"
    assert os.path.exists(args.labels), f"{args.labels} not found"

    print("Reading MNIST data...")
    images = read_idx(args.images)
    labels = read_idx(args.labels)

    if images[1] != labels[1]:
        raise ValueError("Mismatch between number of images and labels")

    print("Connecting to HBase...")
    connection = happybase.Connection(args.zk)
    connection.open()

    ensure_table(connection, args.table)
    print("Uploading to HBase by partitions...")
    put_to_hbase(connection, args.table, images, labels)

if __name__ == "__main__":
    main()
