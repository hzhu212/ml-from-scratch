# 从零构建机器学习算法

参考 [机器学习实战](https://book.douban.com/subject/24703171/) 一书，从零实现了一些基础机器学习算法，包括：

- k近邻算法（KNN）
- 决策树
- 朴素贝叶斯
- 逻辑回归
- AdaBoost元算法
- 线性回归
- 局部加权线性回归
- 岭回归
- 树回归
- k均值聚类（K-Means）
- Apriori关联分析
- 主成分分析（PCA）
- 奇异值分解（SVD）
- 计算图的简单实现

模型的接口设计参照了 keras，主要包括 `fit`、`predict`、`evaluate` 等接口，分别用于模型的训练、预测、评估。

每个数据集均可以通过如下方式载入数据：

```python
from datasets import some_dataset

(x_train, y_train), (x_test, y_test) = some_dataset.load_data()
```
