## 主动学习推荐系统

#### Step1 - 数据预处理

##### 商品信息

- 提取数据集中的title和description信息
- 命令：`python item_information.py [file1, ..., file3]`

##### 用户物品评分信息

- 提取用户-物品评分，划分train集和test集
- 将train集中的用户作为用户全集，以防止出现train集中有用户没有评分的情况
- 命令：`python user_information.py [file1, ..., file7]`

##### 商品相似度生成

- title: 分词 + LDA主题模型（topic number = 15）
- description: 分词 + LDA主题模型（topic number = 15）
- 未使用price（缺失值太多）
- 未使用category（同类商品）
- 命令：`python item_similarity.py [topic number, file1, ..., file4]`

##### 商品description和title相似度权重生成

- non linear regression
- Similarity(i1, i2) = weight1 * S_title(i1) + weight2 * S_description(i2)
- 命令：
  - `python similarity_parameters.py [file1, ..., file7]`
  - `fitnlm(path, param1, param2)`


##### 用户相似度生成

- 评分相似度
- 命令：`python user_similarity.py [file1, ..., file3]`


##### 用户聚类

- 用户聚类依靠用户相似度作为距离度量，使用K-medoids作为聚类算法
- 问题主要存在于：由于评分稀疏，很多用户之间距离为0
- 命令：`python user_clustering.py input_file number_of_clusters output_file`

##### 建树前的准备工作

- 生成用户聚类对任一物品的平均评分，便于计算时直接调用
- 利用非线性回归拟合的参数生成相似度矩阵
- 命令：`python buildtree_preparation.py input_file init_ptitle init_pdescrip output_file`

#### Step2 - 建树及预测

- 树的生成：
  - 三叉树，对应不喜欢、一般般喜欢和喜欢三个节点
  - 生成的节点信息用*self.tree*和*self.node_interval*两个变量保存
- 构建预测模型：
  - 利用Spark的mllib包实现ALS Matrix Factorization
  - 生成伪物品（每个节点）和用户对应的latent vector
- 预测评分：
  - 对每一个test商品，从树的根节点开始向下走，利用目标叶子节点的latent vector作为它的特征向量
  - 利用特征向量和所有物品的特征向量的点积预测评分，计算RMSE
- 命令：`python build_tree.py [input_file1, ..., input_file4] desired_depth`