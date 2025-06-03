import re
from collections import defaultdict

import matplotlib.pyplot as plt

# 读取日志文件内容
with open("output.log", "r", encoding='utf-16') as f:
    log_content = f.read()

# 使用正则表达式提取每个分区的 avg batch loss
pattern = r"Partition\s+(\d+): avg batch loss = ([0-9.]+)"
matches = re.findall(pattern, log_content)

# 使用正则表达式提取 Partition 和 loss
pattern = r"Partition\s+(\d+): avg batch loss = ([0-9.]+)"
matches = re.findall(pattern, log_content)

# 将 loss 按 partition 分类存储
loss_dict = defaultdict(list)

for part_id, loss in matches:
    loss_dict[int(part_id)].append(float(loss))

# 绘图
plt.figure(figsize=(10, 5))

for part_id, losses in sorted(loss_dict.items()):
    steps = list(range(1, len(losses) + 1))
    plt.plot(steps, losses, marker='o', label=f"Partition {part_id}")

plt.title("Avg Batch Loss per Partition")
plt.xlabel("Step")
plt.ylabel("Avg Batch Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
