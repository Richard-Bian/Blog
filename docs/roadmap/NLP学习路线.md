# NLP学习路线

## 1-基础知识

### 线性代数

### 概率论与统计

### 评估指标

## 2-经典模型

### 统计机器学习

- 线性分类

	- 感知机
	- 逻辑回归
	- 朴素贝叶斯

- [SVM](https://xueshu.baidu.com/usercenter/paper/show?paperid=58aa6cfa340e6ae6809c5deadd07d88e&site=xueshu_se)

- 树模型

	- [1984 CART](https://dblp.org/img/paper.dark.empty.16x16.png)

	- [1993 C4.5](https://link.springer.com/article/10.1007/BF00993309)

	- [2001 RF](https://link.springer.com/article/10.1023%2FA%3A1010933404324)

	- [2016 XGBoost](https://dl.acm.org/doi/10.1145/2939672.2939785)

	- [2017 LightGBM](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

- HMM/CRF

### 神经网络

- 词向量

	- Word2Vec
	- GloVe
	- Fasttext

- CNN
- RNN/LSTM/GRU/Bidirectional
- [ELMo](https://doi.org/10.18653/v1/n18-1202)

- BERT

## 3-任务范式与技巧

### 文本分类

A Survey on Text Classification: From Shallow to Deep Learning
https://arxiv.org/abs/2008.00364

- ReNN

	- [2011 RAE](http://ai.stanford.edu/~ang/papers/emnlp11-RecursiveAutoencodersSentimentDistributions.pdf)

	- [2012 MV-RNN](https://ai.stanford.edu/~ang/papers/emnlp12-SemanticCompositionalityRecursiveMatrixVectorSpaces.pdf)

	- [2013 RNTN](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)

	- [2014 DeepRNN](https://papers.nips.cc/paper/5275-global-belief-recursive-neural-networks.pdf)

- MLP

	- [2014 Paragraph-Vec](http://proceedings.mlr.press/v32/le14.html)

	- [2015 DAN](https://doi.org/10.3115/v1/p15-1162)

- RNN

	- [2015 TreeLSTM](https://doi.org/10.3115/v1/p15-1150)

	- [2015 S-LSTM](http://proceedings.mlr.press/v37/zhub15.pdf)

	- [2015 TextRCNN](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745)

	- [2015 MT-LSTM](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.1430&rep=rep1&type=pdf)

	- [2016 oh-2LSTMp](https://www.researchgate.net/publication/303521296_Adversarial_Training_Methods_for_Semi-Supervised_Text_Classification)

	- [2016 BLSTM-2DCNN](https://www.aclweb.org/anthology/C16-1329.pdf)

	- [2017 DeepMoji](https://www.aclweb.org/anthology/D17-1169.pdf)

	- [2017 TopicRNN](https://openreview.net/forum?id=rJbbOLcex)

	- [2017 Miyato et al.](https://arxiv.org/abs/1605.07725)

	- [2018 RNN-Capsule](https://link.springer.com/article/10.1007/s42979-020-0076-y)

	- 2018 ELMo

- CNN

	- [2014 TextCNN](https://www.aclweb.org/anthology/D14-1181.pdf)

	- [2014 DCNN](https://doi.org/10.3115/v1/p14-1062)

	- [2015 CharCNN](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification)

	- [2016 SeqTextRCNN](https://arxiv.org/abs/1603.03827)

	- [2017 XML-CNN](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)

	- [2017 DPCNN](https://doi.org/10.18653/v1/P17-1052)

	- [2017 KPCNN](https://www.ijcai.org/Proceedings/2017/0406.pdf)

	- [2018 TextCapsule](https://doi.org/10.18653/v1/d18-1350)

	- [2018 HFT-CNN](https://www.aclweb.org/anthology/D18-1093.pdf)

	- [2020 Bao et al.](https://arxiv.org/abs/1908.06039v1)

- Attention

	- [2016 HAN](https://doi.org/10.18653/v1/n16-1174)

	- [2016 BI-Attention](https://www.aclweb.org/anthology/D16-1024.pdf)

	- [2016 LSTMN](https://doi.org/10.18653/v1/d16-1053)

	- [2017 Lin et al.](https://arxiv.org/abs/1703.03130)

	- [2018 SGM](https://www.aclweb.org/anthology/C18-1330/)

	- [2018 BiBloSA](https://arxiv.org/abs/1804.00857)

	- [2019 AttentionXML](https://arxiv.org/pdf/1811.01727v3.pdf)

	- [2019 HAPN](https://www.aclweb.org/anthology/D19-1045/)

	- [2019 Proto-HATT](https://gaotianyu1350.github.io/assets/aaai2019_hatt_paper.pdf)

	- [2019 STCKA](https://arxiv.org/pdf/1902.08050.pdf)

- Transformer

	- 见预训练

- GNN

	- [2018 DGCNN](https://dl.acm.org/doi/10.1145/3178876.3186005)

    - [2018 GAT](https://mila.quebec/wp-content/uploads/2018/07/d1ac95b60310f43bb5a0b8024522fbe08fb2a482.pdf)

	- [2019 TextGCN](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4725)

	- [2019 SGC](https://arxiv.org/pdf/1902.07153.pdf)

	- [2019 Huang et al.](https://www.aclweb.org/anthology/D19-1345.pdf)

	- [2019 Peng et al.](https://arxiv.org/abs/1906.04898)

	- [2020 MAGNET](https://arxiv.org/abs/2003.11644)

### 句间匹配

- Representation-based

	- 监督

		- 2013 DSSM
		- 2015 SiamCNN
		- 2016 Multi-view
		- 2016 SiamLSTM

	- 无监督

		- 词向量

			- 2016 WMD
			- 2017 SIF

		- 句向量

			- 2015 SkipThought
			- 2016 FastSent

	- 监督+迁移

		- 2017 Joint-many
		- 2017 InferSent
		- 2017 SSE 
		- 2018 GenSen
		- 2018 USE
		- 2018 MT-DAN(USE)
		- 2019 Sentence-BERT
		- 2020 BERT-flow

- Interaction-based

	- 2016 DecAtt
	- 2016 PWIM
	- 2016 MatchCNN
	- 2017 ESIM
	- 2018 DIIN
	- 2018 BERT
	- 2019 HCAN
	- 2019 RE2

### 序列标注

A Survey on Recent Advances in Sequence 
Labeling from Deep Learning Models 
https://arxiv.org/pdf/2011.06727

- Embedding Module

	- Hand-crafted Features

		- Spelling features

			- [2011 Word suffix, gazetteer, capitalization, tags](https://jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

		- Context features

			- [2015 unigram, bi-gram, tri-gram](https://arxiv.org/pdf/1508.01991v1.pdf)

	- Pretrained Word Embeddings

		- [2011 Senna](https://jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

		- Word2Vec
		- GloVe
		- ELMo
		- BERT

	- Character-level Representations

	  解决OOV问题
	  利用word morphological和shape信息
	  
		- CNN

			- [2014 Max pooling](http://proceedings.mlr.press/v32/santos14.pdf)

			- [2018 IntNet](https://arxiv.org/abs/1810.12443v1)

		- RNN

			- [2015 BiLSTM on word+Final States](http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf)

			- [2017 +Attention](https://nlp.stanford.edu/pubs/dozat2017stanford.pdf)

			- [2018 +Self-supervised Task](https://www.aclweb.org/anthology/W18-3401.pdf)

			- [2018 BiLSTM on sentence+Word Final States](https://arxiv.org/abs/1805.08237)

	- Sentence-level Representations

		- [2019 Assign every word with a global representation](https://www.aclweb.org/anthology/P19-1233.pdf)

- Context Encoder Module

	- RNN

		- [2015 Bi-LSTM](https://arxiv.org/pdf/1508.01991v1.pdf)

		- [2017 Bi-GRU](https://www.ijcai.org/Proceedings/2018/0579.pdf)

		- [2017 LM as auxiliary-task](https://www.aclweb.org/anthology/P17-1194.pdf)

		- [2017 Multi-Order BiLSTM](https://www.aclweb.org/anthology/C18-1061.pdf)

		- [2017 Classification as auxiliary-task](https://arxiv.org/pdf/1709.10191.pdf)

		- [2018 Parallel BiLSTM](https://www.aclweb.org/anthology/P18-2012.pdf)

		- [2017 Implicitly-defined neural networks](https://www.aclweb.org/anthology/P17-2027/)

		- [2018 Lattice-LSTM](https://arxiv.org/abs/1805.02023)

		- [2019 Deep-transition RNN](https://arxiv.org/pdf/1312.6026.pdf)

		- [2020 +Attention](https://www.sciencedirect.com/science/article/pii/S0031320320304398)

	- [CNN](https://jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

		- [2017 GCNN](http://www.cips-cl.org/static/anthology/CCL-2017/CCL-17-071.pdf)

		- [2017 ID-CNN](https://arxiv.org/abs/1702.02098)

		- [2019 LR-CNN](https://link.zhihu.com/?target=https%3A//pdfs.semanticscholar.org/1698/d96c6fffee9ec969e07a58bab62cb4836614.pdf)

		- [2019 GRN](https://arxiv.org/pdf/1907.05611v2.pdf)

	- GNN

		- [2019 CGN](https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1396.pdf)

	- Transformer

		- [2019 TENER](https://arxiv.org/abs/1911.04474)

		- [2019 Star-Transformer](https://arxiv.org/abs/1902.09113)

		- [2020 FLAT](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2004.11795.pdf)

- Inference Module

	- Softmax
	- CRF

		- [2005 Semi-CRF](http://www.cs.cmu.edu/~wcohen/postscript/semiCRF.pdf)

			- [2015 SRNN](https://arxiv.org/abs/1511.06018)

			- [2016 grSemi-CRFs](http://ml.cs.tsinghua.edu.cn/~jun/pub/semi-CRF-acl2016.pdf)

			- [2018 HSCRF](https://www.aclweb.org/anthology/P18-2038.pdf)

		- [2012 Skip-chain CRF](https://arxiv.org/pdf/1011.4088v1.pdf)

		- [2018 Embedded-State Latent CRF](https://arxiv.org/abs/1809.10835)

		- [2019 NCRF transducers](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ic19-NCRFT.pdf)

	- RNN

		- [2016 parallel with context encoder](https://yonatanbisk.com/papers/2016-NAACLShort.pdf)

		- [2018 before Softmax](https://arxiv.org/abs/1707.05928)

		- [2018 joint labeling](https://www.ijcai.org/Proceedings/2018/0637.pdf)

		- [2017 Encoder-Decoder-Pointer Framework](https://arxiv.org/pdf/1701.04027.pdf)

### 文本生成

2017 Neural text generation: A practical guide 

2018 Neural Text Generation: Past, Present and Beyond

2019 The survey: Text generation models in deep learning

https://github.com/pytorch/fairseq

https://zhuanlan.zhihu.com/p/162035103

### 语言模型

- word-level

	- generative

		- MLM

			- Subword (BERT)

			  15% tokens: 80% [MASK], 10% random, 10% unchanged
			  在预处理阶段，给每个句子采样10种mask方式
			- Word (WWM)

			  根据分词后的词边界mask
			  
			  Pre-Training with Whole Word Masking for Chinese BERT
			  2019.6.19
			  https://arxiv.org/abs/1906.08101
			- Entity/Phrase (Baidu ERNIE1.0)

			  Mask策略：word-level/phrase-leve/entity-level
			  50%的时候选entity或phrase，剩下选word（保持总体subword在15%）
			  1.0: https://arxiv.org/abs/1904.09223
			  2019.4.19
			  2.0: https://arxiv.org/abs/1907.12412
			  2019.7.29
			- Span/N-gram (SpanBERT)

			  根据几何分布，先随机选择一段（span）的长度，之后再根据均匀分布随机选择这一段的起始位置，最后按照长度遮盖。文中使用几何分布取 p=0.2，最大长度只能是 10，平均被遮盖长度是 3.8 个词的长度。
			  参考Roberta的动态masking/一直训练长句
			  2019.7.24
			  https://arxiv.org/abs/1907.10529
			  https://zhuanlan.zhihu.com/p/75893972
			- Dynamic (RoBERTa)

			  每个Epoch见到的样本mask位置都不一样，实际上效果只提升了不到一个点
			  
			  RoBERTa: A Robustly Optimized BERT Pretraining Approach
			  2019.7.26
			  https://arxiv.org/abs/1907.11692
		- PLM (XLNet)

		  XLNet: Generalized Autoregressive Pretraining for Language Understanding
		  2019.6.19
		  https://arxiv.org/abs/1906.08237
		  https://zhuanlan.zhihu.com/p/70218096
		- SBO (SpanBERT)

		  在训练时取 Span 前后边界的两个词，不在 Span 内，然后用这两个词向量加上 Span 中被遮盖掉词的位置向量，来预测原词。详细做法是将词向量和位置向量拼接起来，过两层全连接层
		  比NSP表现好，有一个点的提升（个别3个点）
		  在span抽取式任务上有很大提升
		- InfoWord

		  ICLR2020
		  A Mutual Information Maximization Perspective of Language Representation Learning
		  2019.10.18
		  DeepMind & CMU
		  https://arxiv.org/abs/1910.08350
	- discrimitive

		- WSO (StructBERT)

		  Word Structural Objective
		  按K个一组打乱token顺序，预测原顺序（5%个trigram）
		  和MLM jointly等权重训练
		  
		  平均不到1个点或负增长，CoLA任务上有4个点的提升
		  
		  ICLR2020
		  https://arxiv.org/abs/1908.04577
		  
		  问题：
		  1. 负增长是否由于joint训练？mask掉和需要预测位置的重合？以前只有15%的噪音，现在有30%
		  2. pretrain batchsize
		- RTD (ELECTRA)

		  ICLR2020
		  https://openreview.net/forum?id=r1xMH1BtvB
		  https://zhuanlan.zhihu.com/p/89763176
		- Capitalization Prediction (ERNIE2.0)

		  判断token是否大写，对英文NER有用
		- Token-Document Relation (ERNIE2.0)

		  判断token是否在文中其他地方出现
		  作者认为重复出现的都是比较重要的词
	
- sentence-level

	- self-supervised

		- NSP (BERT)

		  2 class: 50% next, 50% random from corpus
		  从消融实验来看，只对QNLI任务影响较大（3.5），对其他任务只有不到1%对影响
		  
		  缺点：
		  SpanBERT：
		  1.相比起两句拼接，一句长句，模型可以获得更长上下文（类似 XLNet 的一部分效果）；
		  2.在 NSP 的负例情况下，基于另一个文档的句子来预测词，会给 MLM 任务带来很大噪音。
		  ALBERT：
		  3.学到的是主题相关
		  RoBERTa：
		  4. BERT的消融实验可能只去掉了NSP的loss term，但输入仍是sentence pair
		  Symmetric Regularization：
		  5. BERT的顺序敏感，致使swap之后NLI任务效果下降
		- NSP+SOP (StructBERT)

		  3 class：预测是上一句/下一句/随机
		  
		  平均不到1个点的提升
		  
		  StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding
		  2019.8.13
		  https://arxiv.org/abs/1908.04577
		- PN5cls+smth

		  previous sentence prediction
		  
		  5 classs: next/prev/next-inadjacent/prev-inadjacent/diffdoc
		  在prev-inadj/next-inadj上用了0.8的label smoothing到prev/next
		  比BERTbase提升约1个点（8个glue任务）
		  
		  Symmetric Regularization based BERT for Pair-wise Semantic Reasoning
		  2019.9.8
		  蚂蚁金服+达摩院
		  https://arxiv.org/abs/1909.03405
		- SOP (ALBERT)

		  Sentence Order Prediction
		  2class: 是next=1，是prev=0
		  提升1-3个点
		  
		  ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
		  ICLR2020
		  2019.9.26
		  Google+Toyota
		  https://arxiv.org/abs/1909.11942
		  https://zhuanlan.zhihu.com/p/84273154
		- Sentence Reordering (ERNIE2.0)

		  把一段中的句子划分为m个片段，打乱，进行K分类
		  K = sum(n!), n = 1, .., m
		  
		  2019.7.29
		- Sentence Distance (ERNIE2.0)

		  3 class：在同一篇文档且相邻/在同一篇文档不相邻/不在同一篇文档
		
	- supervised
	
		- DLM (Baidu ERNIE1.0)
	
		  Dialogue Language Model
		  多轮对话：QRQ, QRR, QQR
		  2 class: 判断对话是真实的还是Fake，和MLM任务交替训练
		  有一个点的提升
		  
		- IR Relevance (ERNIE2.0)
		
		  3 class: 被点击/出现在搜索结果中/随机
		  
		- Discourse Relation (ERNIE2.0)
		判断句子的语义关系例如logical relationship( is a, has a, contract etc.)

## 4-应用场景

### 单一任务

数据集
评估方式

- 词法分析

	- 分词
	- 词性识别
	- NER

- 句法分析

	- 依存分析
	- 语义角色标注

- 语义分析

	- 情感分析
	- 意图识别
	- 信息抽取
	- 同义识别
	- 指代消解
	- 阅读理解
	- 文本纠错

- 文本生成

	- 生成式摘要
	- 机器翻译
	- 对话问答

### 复杂任务

- 搜索/推荐
- 对话
- 知识图谱构建/应用

