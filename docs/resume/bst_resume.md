## **边圣陶**

<div>
<img align=right src="https://cdn.jsdelivr.net/gh/Richard-Bian/imagebed@main/uPic/%E6%8A%A5%E5%90%8D%E7%85%A7%E7%89%87.jpg" width="100" height="125" />
</div>
- 出生年月：1995-06

## 联系方式

- 手机：15557537168

- Email：[bian.shengtao@gmail.com](mailto:bian.shengtao@gmail.com)
- 微信号：15557537168
## 教育经历
- **硕士  人工智能与数据科学  工程学院   新南威尔士大学**   
  2020.09 ～2023.02 

- **本科  电子商务  法学（双学位）  信息管理学院  山西财经大学**
  2014.09～2018.07

## 实习经历

**京东集团股份有限公司 
NLP算法实习生 
起止时间：2022.5～至今**

1. 负责用tensorflow2复现simCSE模型。将动态mask的MLM任务与simCSE结合，并改写tensorflow2的    SparseCategoricalCrossentropy损失函数，实现mlm损失函数和simCSE损失函数共同计算。
2. 负责用京东内部数据对simCSE进行微调，并用TF Serving部署上线。
3. 参与对京东产品的迭代，优化神经检索模型，改写内部相似度匹配机制和类目推荐机制。把原来的Sbert替换成simCSE；将内部相似度匹配机制从欧氏距离改为余弦相似度；类目向量由原来的类目下的产品通过kmeans聚合得到，转变为将类目名称直接通过simCSE得到其embedding。将检索准确率从76%优化到97%。
4. 总负责建立图像&视频-文本的跨模态检索产品的PoC。建立图片视频的混合素材库；通过CLIP模型对图片直接进行embedding，对视频进行抽帧，进而对抽出来的图片进行embedding。将所有embedding保存在faiss中，建立整个素材库的索引。通过用CLIP模型对文本进行embedding，在faiss中搜索相关图片和视频相关片段并返回。
5. 负责整理微调CLIP模型的电商数据。抽取京东商城数据库商品的图文对，对商品名称进行prompt，整理成CLIP的训练集。

**杭州海康威视数字技术股份有限公司
NLP算法实习生 
起止时间：2022.02～2022.05**

1. 研究金融领域事件抽取方向论文，如Doc2EDAG，GIT等，并撰写算法综述。
2. 研究长文本表征的方法，如roformer，层次分解位置编码等让bert突破512个字符限制的方法，整理代码并撰写算法综述。
3. 研究并整理文本句间匹配方向的算法方向，如Sbert，simCSE，esimCSE等，整理代码并撰写算法综述。

**树根互联信息科技有限公司（三一重工）
	
起止时间：2021.10～2022.01**

1. 负责整理清洗问答对，并录入数据库。
2. 参与建立搜索功能，用elasticsearch库建立问答对的倒排索引。用elasticsearch做短语层面的相似性计算。
3. 用simbert将题库中的标准问题和用户提问转化成embedding，并用faiss存储。并对用户提问通过余弦相似性计算获得相近语义的标准问题。
4. 统合上述两者返回的topk以及score，对超过相似阈值的则返回问答库中的标准问题和回答，并对相似度低的问题录入陌生问题数据库中。
5. 用自产数据对原有simbert预训练模型继续fintune。
6. 对自研数据进行实体和关系抽取，并整理。
7. 将抽取的实体和关系存入Neo4j中，建立知识图谱。

## 项目经历

- **基于Jina和ffmpeg的 Video Clip Extraction by description**
  起止时间：2022.03 ~ 2022.05

  **项目描述**
  此项目为DataWhale开源社区和Jina.ai合作项目。基于streamlit的框架搭建前端，基于Jina的神经网络搜索框架搭建后端，部署CLIP模型，建立神经检索机制。对上传的视频用ffmpeg进行抽帧，通过CLIP对图片进行embedding，实现通过文字描述定位视频的关键帧，映射到视频的start时间节点和end时间节点，剪辑视频并返回。

- **自然语言理解进阶探索**
  起止时间：2022.01 ~ 至今

  **项目描述**
  此项目为学校导师paper的一部分，paper在投。基于bert-style预训练模型(Roberta,Deberta)，Dataset: Multi-genre natural language inference (MNLI) dataset ，对前沿sentence embedding做进阶探索。目前主要工作是通过discriminative loss(用于预测ground-truth NLP labels)和adversarial loss(用于discriminator在NLI的三个标签上产生均匀分布)来构建设计一种adversarial learning strategy。

## 比赛经历

- **Predicting Eligibility for the Emergency Broadband Benefit Program**
  起止时间：2022.03 ~ 2022.04
  **赛事链接**：https://trachack.com/challenge-22-2/

  **赛事背景**
  赛事由新南威尔士大学，迈阿密大学，纳瓦拉大学联合参与，旨在使用机器学习预测哪些用户有资格获得紧急宽带福利资格(EBB)而目前没有使用，使得联邦通信委员会 (FCC)可以主动联系他们参与该计划。该比赛所给数据集为10多个CSV数据，涵盖用户身份信息，通信信息，消费层级，等等一系列feature，所给样本在15万条以上，正负样本比例极不均衡。方案的准确性根据F1-socre进行评估。模型限制为非深度学习模型。

  **比赛名次和所做工作**

  本次比赛所在小队获得第二名(共60多支队伍)，奖金为$3000。最终F1-score得分为0.997。本人在项目小队中所负责的工作有：

  1. 导入数据，清洗数据并合并数据，将各类特征进行分类。
  2. 进行特征工程，将文本数据统一（例如将Apple 和 APPLE统一）并映射为数字。根据最早时间处理时间戳为时长，并分离为多个维度（年月日）。对年龄，所在区域等features进行分箱。生成交叉特征（例如手机品牌和运营商）等。
  3. 选择IsolationForest，LocalOutlierFactor，KNN作为此次比赛分类器，用RandomizedSearch进行参数调优。用validset得到验证结果并上传。

- **新冠疫情相似句对判定大赛**
  起止时间：2020.02 ~ 2020.04
  **赛事链接：**https://tianchi.aliyun.com/competition/entrance/231776/introduction

  **赛事背景**
  赛事由天池大赛举办，旨在助力疫情智能问答应用技术精准度提升，探索下一代医疗智能问答技术。该比赛所给数据集每一条数据由Id, Category，Query1，Query2，Label构成，分别表示问题编号、类别、问句1、问句2、标签。Label表示问句之间的语义是否相同，若相同，标为1，若不相同，标为0。方案的准确性根据Accuracy进行评估。

  **比赛名次和所做工作**
  本次比赛所在小队获得第27名(共900多支队伍)。最终Accuracy得分为0.9582(与第一名差0.0062)。本人在项目小队中所负责的工作有：

  1. 通过传递相似性，实体替换等方式对数据进行增强。
  2. 用TextCNN，BERT+MLP，SBERT等模型的进行文本匹配，并在分割出来的测试集上进行测试。此处在RoBERTa_wwm_ext_large的基础上进行fine-tune，并在最后一层增加了一层Dense层进行文本匹配。另外通过对SBERT的微调，分别计算句一和句二的sentence embedding并计算相似度进行文本匹配。

## 技能清单

- 编程语言：Python/C++/SQL/CYPHER
- 机器学习框架：PyTorch/Tensorflow/Keras/Scikit-learn
- Web框架：Flask/fastapi
- 数据库相关：MySQL/PgSQL
- No-SQL：Redis/mongodb/neo4j
- 版本管理、文档和自动化部署工具：Git/docker



1. responsible for replicating the simCSE model with tensorflow2. Combine the MLM task of dynamic mask with simCSE, and rewrite the SparseCategoricalCrossentropy loss function of tensorflow2 to realize the joint calculation of mlm loss function and simCSE loss function.
2. Responsible for fine-tuning simCSE with Jingdong internal data and deploying it online with TF Serving.
3. Participate in the iteration of Jingdong products, optimize the neural retrieval model, rewrite the internal similarity matching mechanism and category recommendation mechanism. Replace the original Sbert with simCSE; change the internal similarity matching mechanism from Euclidean distance to cosine similarity; change the category vector from the original product under the category by kmeans aggregation to the category name directly by simCSE to get its embedding. optimize the retrieval accuracy from 76% to 97%.
4. responsible for the feasibility verification of building cross-modal retrieval products of image & video-text. Build a mixed material library of images and videos; embedding images directly by CLIP model, drawing frames for videos, and then embedding the drawn images. save all embeddings in faiss and build the index of the whole material library. By embedding the text with the CLIP model, search for relevant images and video related clips in faiss and return them.
5. Responsible for finishing fine-tuning the e-commerce data of CLIP model. Extract the image and text pairs of Jingdong Mall database products, prompt the product names, and organize them into the training set of CLIP.





1. research event extraction direction papers in finance, such as Doc2EDAG, GIT, etc., and write algorithm reviews.
2. research methods of long text representation, such as roformer, hierarchical decomposition position coding, etc. to let bert break the 512 character limit, organize the code and write algorithm reviews.
3. research and organize the algorithm direction of text inter-sentence matching, such as Sbert, simCSE, esimCSE, etc., organize the code and write the algorithm review.





1. Responsible for collating and cleaning Q&A pairs and entering them into the database.
2. Participate in building search function and build inverted index of Q&A pairs with elasticsearch library. Use elasticsearch to do similarity calculation at phrase level.
3. use simbert to transform the standard questions and user questions in the question database into embedding and store them with faiss. And get the standard questions with similar semantics by cosine similarity calculation for user questions.
4. Combine the topk and score returned by the above two, and return the standard questions and answers in the question and answer database for those exceeding the similarity threshold, and record the questions with low similarity into the database of unfamiliar questions.
5. continue to fintune the original simbert pre-training model with self-produced data.
6. Perform entity and relationship extraction on the self-generated data and organize them.
7. Store the extracted entities and relations into Neo4j to build the knowledge graph.
