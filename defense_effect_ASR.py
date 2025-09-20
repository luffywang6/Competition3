import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
categories = ["AIDS", "NCI1", "MCF-7", "Mutagenicity"]  # 分组类别
values0 = [100, 100, 100, 100]
values1 = [1.82, 10.35, 0, 13.33]

# 设置柱的宽度和间隔
bar_width = 0.3  # 柱宽度
node_labels = np.arange(len(values0))
indexes = np.arange(len(values0))  # 分组索引

plt.figure(figsize=(10, 5))

plt.bar(indexes, values0, bar_width, label="ASR before defense", color="blue")
plt.bar(indexes + bar_width, values1, bar_width, label="ASR after defense", color="green")

# 设置轴标签、标题和图例
plt.xlabel("dataset")
plt.ylabel("%", rotation=0)
# plt.ylim(0, 1)
plt.title("ASR")
plt.xticks(indexes + bar_width / 2, categories)
plt.legend(loc="upper right")

# 显示图形
plt.show()
