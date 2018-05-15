## 主动学习推荐系统

### 0. 概述

- 数据集：[Amazon商品数据集](http://jmcauley.ucsd.edu/data/amazon/)
- 编程环境：Python, Matlab, Markdown

### 1. 数据预处理

- 商品信息
  - 提取数据集中的title和description信息
  - 命令：`python item_information.py [file1, ..., file3]`

- 用户物品评分信息

  - 提取用户-物品评分，划分train集和test集
  - 将train集中的用户作为用户全集，以防止出现train集中有用户没有评分的情况
  - 命令：`python user_information.py [file1, ..., file7]`

- 商品相似度生成

  - title: 分词 + LDA主题模型（topic number = 15）
  - description: 分词 + LDA主题模型（topic number = 15）
  - 未使用price（缺失值太多）
  - 未使用category（同类商品）
  - 命令：`python item_similarity.py [topic number, file1, ..., file6]`

- 商品description和title相似度权重生成

  - non linear regression
  - Similarity(i1, i2) = weight1 * S_title(i1) + weight2 * S_description(i2)
  - 命令：
    - `python similarity_parameters.py [file1, ..., file7]`
    - `fitnlm(path, param1, param2)`

- 用户相似度生成

  - 评分相似度
  - 命令：`python user_similarity.py [file1, ..., file3]`

- 用户聚类

  - 用户聚类依靠用户相似度作为距离度量，使用K-medoids作为聚类算法
  - 问题主要存在于：由于评分稀疏，很多用户之间距离为0
  - 命令：`python user_clustering.py input_file number_of_clusters output_file`

- 建树前的准备工作

  - 生成用户聚类对任一物品的平均评分，便于计算时直接调用
  - 利用非线性回归拟合的参数生成相似度矩阵
  - 命令：`python buildtree_preparation.py input_file init_ptitle init_pdescrip output_file`

### 2. 建树及预测

- 树的生成：
  - 三叉树，对应不喜欢、一般般喜欢和喜欢三个节点
  - 生成的节点信息用*self.tree*和*self.node_interval*两个变量保存
- 构建预测模型：
  - 利用Spark的mllib包实现ALS Matrix Factorization
  - 生成伪物品（每个节点）和用户对应的latent vector（对每一层都计算）
- 预测评分：
  - 对每一个test商品，从树的根节点开始向下走，利用目标叶子节点的latent vector作为它的特征向量
  - 利用特征向量和所有物品的特征向量的点积预测评分，计算RMSE（对每一层都计算）
- 命令：`python build_tree.py [input_file1, ..., input_file5] desired_depth`

### 3. 运行

- 利用*Python*脚本运行上述所有步骤：`python script.py`
- 代码开头数据集名称（*dataset*）需相应更改

### 4. 对比实验

- FDT (Factorized Deicision Tree)
  - `python factorized_decision_tree.py dataset depth`  (dataset是数据集的名字，depth决定了树的高度)
  - **输入：** *I\*U* 的矩阵 => *new-user problem*
  - **输入：** *U\*I* 的矩阵 => *new-item problem*
- CAL (Content-based Active Learning)
	- `python content_based_active_learning.py dataset K`  (dataset是数据集的名字，K决定了选择TopK的用户进行query)
- CBCF (Content-based Collaborative Filtering)

### 4. 当前问题

- 对All_Beauty数据集来说树的第一层预测效果最好，分析原因可能如下：
  - 数据集过于稀疏（0.02%），导致每一用户基本只有一个评分，第一层作为伪物品作矩阵分解时评分满，效果好，越往下效果越差。
  - 点的划分过于不均匀，使得伪物品选择不优秀，可试平均划分。
- 物品个数超过30万的Automotive集合上计算*item similarity*时出现*Memory Error*
  - 已解决，选择评分个数大于5个的物品和用户