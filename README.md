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
- 命令：`python item_similarity.py [topic number, file1, ..., file8]`

