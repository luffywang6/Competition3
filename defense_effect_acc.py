import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
categories = ["AIDS", "NCI1", "MCF-7", "Mutagenicity"]  # 分组类别
values0 = [99.25, 67.03, 91.74, 71.08]
values1 = [97.17, 62.62, 91.50, 73.18]

# 设置柱的宽度和间隔
bar_width = 0.3  # 柱宽度
node_labels = np.arange(len(values0))
indexes = np.arange(len(values0))  # 分组索引

plt.figure(figsize=(10, 5))

plt.bar(indexes, values0, bar_width, label="clean accuracy", color="blue")
plt.bar(indexes + bar_width, values1, bar_width, label="defense accuracy", color="green")

# 设置轴标签、标题和图例
plt.xlabel("dataset")
plt.ylabel("%", rotation=0)
# plt.ylim(0, 1)
plt.title("accuracy")
plt.xticks(indexes + bar_width / 2, categories)
plt.legend(loc="upper right")

# 显示图形
plt.show()
