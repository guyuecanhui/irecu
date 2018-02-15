# SVD++

> 论文阅读——Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model

本文对协同过滤中最主要的两种方法（基于邻域的方法和基于隐特征模型的方法）分别提出了优化方案，并且设计了一个联合模型将两种方法统一，从而达到更好的效果。为了进行区分，本文将对SVD进行优化的方案称为SVD+，将联合模型的方法称为SVD++。

---

## 一、研究背景

Koren在做Netflix的比赛过程中，发现基于邻域的方法和基于隐特征模型的方法各有所长：

|比较|基于邻域的方法|基于隐特征模型的方法|
|---|---|---|
|主要思想|核心在于计算用户/物品的相似度，将相似用户的喜好推荐给用户，或将用户喜欢物品的相似仿物品推荐给用户|假设真正描述用户评分矩阵性质的内存特征（可能未知）其实只有少数几个，将用户和物品都映射到这些隐特征层，从而使得用户和物品直接关联起来|
|挖掘信息特征|能够对局部强相关的关系更敏感，而无法捕捉全局弱相关的关系|能够估计关联所有物品/用户的整体结构，但是难以反映局部强相关的关系|

因此，这两种方法存在天然的互补关系。另外，Koren还发现，使用隐式反馈的数据能够提高推荐的准确性，而这两种方法都不支持使用隐式反馈的数据。基于这些发现，Koren先分别将隐式反馈集成到两个模型中去，得到两个优化的模型，再提出一种联合模型，将这两个优化的模型进一步融合，从而得到更好的效果。

## 二、模型推导

文章从Baseline的模型，通过加入各种考虑因素，推导出基于邻域和基于隐特征的两个模型，再推导出联合模型。

### 2.1、Baseline模型

Baseline模型就是基于历史数据的简单统计，主要看用户 $$u$$ 的平均评分 $$b_u$$、电影 $$i$$ 的平均评分 $$b_i$$ 和所有电影的平均评分 $$\mu$$：
$$
b_{ui} = \mu + b_u + b_i
$$
所有后面的模型都是对这个基准模型的修正。这个基准模型中的参数都是可以离线计算的，用的方法也是本文通用的参数估计方法，先定义损失函数 $$l(P)$$：

$$
l(p_1,p_2,\cdots) = \sum_{(u,i)\in \kappa} (r_{ui} - \hat{r_{ui}})^2 + \lambda(\sum_{p_1} p_1^2 + \sum_{p_2} p_2^2 + \cdots)
$$

其中，$$P={p_1,p_2,\cdots}$$ 表示待估计的参数，$$\kappa$$ 表示所有显式反馈的组合（即用户 $$u$$ 对物品 $$i$$ 进行过评分），$$r_{ui}$$ 表示评分的实际值，$$\hat{r_{ui}}$$ 表示评分的预测值，$$\lambda$$ 为超参，根据经验设置，然后求最小化 $$l(P)$$ 下各参数的值，通常使用最小二乘法，或者文中使用的梯度下降法（效率更高）。比如这个地方，参数就 $$b_u$$ 和 $$b_i$$，可以根据下式进行参数估计：

$$
\min_{b_*}\sum_{(u,i)\in \kappa} (r_{ui} - \hat{r_{ui}})^2 + \lambda(\sum_{p_1} p_1^2 + \sum_{p_2} p_2^2 + \cdots)
$$

### 2.2、推广到基于邻域的模型

本文主要考虑ItemCF，对于两个物品$$i$$和$$j$$，它们的相似性$$s_{ij}$$是基于Pearson相关系数$$\rho_{ij}$$计算得到：

$$
s_{ij} = \frac{n_{ij}}{n_{ij}+\lambda_2}\rho_{ij}, \ \ \rho_{ij}=\frac{E((x-\mu_x)(y-\mu_y))}{\sigma_x\sigma_y}
$$

其中，$$n_{ij}$$表示同时对$$i$$和$$j$$进行评分的用户数，$$\lambda_2$$应该是防止$$i$$和$$j$$比较冷门的情况下，恰好有个别用户同时对它们进行了评分，这时候它们的相关性实际是看不出来的，属于偶然情况，通常$$\lambda_2=100$$。之前的ItemCF进一步利用用户$$u$$评过分的与$$i$$最相关的$$k$$个物品$$S^k(i;u)$$来估计用户$$u$$对$$i$$的评分：

$$
\hat{r_{ui}} = b_{ui} + \frac{\sum_{j\in S^k(i;u)} s_{ij}(r_{uj} - b_{uj})}{\sum_{j\in S^k(i;u)} s_{ij}}
$$

但是如果$$u$$没有对与$$i$$相似的物品评过分，那上式就主要取决于$$b_{ui}$$了。为了解决这个小问题，有方案先计算插值权重$$\theta_{ij}^u$$来取代实际的评分：

$$
\hat{r_{ui}} = b_{ui} + \sum_{j\in S^k(i;u)} \theta_{ij}^u (r_{uj} - b_{uj})
$$

但是以上模型都只考虑了用户$$u$$，而对全局结构没有一个很好的理解，因此Koren提出不仅仅使用用户$$u$$的对$$i$$最相关的$$k$$个物品的评分数据，而是使用所有$$u$$的评分数据，因此引入一个参数$$\omega_{ij}$$来表示$$j$$的评分对$$i$$评分的影响，并且这个$$\omega_{ij}$$是基于所有用户对$$i$$和$$j$$评分估计出来的：

$$
\hat{r_{ui}} = b_{ui} + \sum_{j\in R(u)} (r_{uj} - b_{uj})\omega_{ij}
$$

分析这个式子，当$$i$$和$$j$$越相关，说明$$j$$对$$i$$的影响越大，即$$w_{ij}$$越大，这时候如果$$(r_{uj} - b_{uj})$$较大，则估计的评分相对于$$b_{ui}$$的偏移也就越多；反之，当$$w_{ij}$$较小时，无论$$j$$的评分如何都对偏移影响不大。

在此基础上，进一步引入隐式反馈的数据：

$$
\hat{r_{ui}} = b_{ui} + \sum_{j\in R(u)} (r_{uj} - b_{uj})\omega_{ij} +\sum_{j\in N(u)} c_{ij}
$$

其中，$$c_{ij}$$表示隐式反馈对基准估计的偏移影响，当$$j$$与$$i$$的评分强相关时，$$c_{ij}$$较大。这个式子的主要问题是，它对重度用户的推荐和对轻度用户的推荐结果相差较大，因为重度用户的显式反馈和隐式反馈都很多，因此偏移项值较大。Koren发现，做一下规范化以后，效果会更好：

$$
\hat{r_{ui}} = b_{ui} + \mid R(u)\mid ^{-1/2}\sum_{j\in R(u)} (r_{uj} - b_{uj})\omega_{ij} +\mid N(u)\mid ^{-1/2}\sum_{j\in N(u)} c_{ij}
$$

为了降低上式的计算复杂度，可以只考虑对$$i$$影响最大的$$k$$个物品，记$$R^k(i;u)=R(u)\cap S^k(i)$$表示$$u$$评分过的物品中属于$$i$$最相似的Top k物品，类似的，记$$N^k(i;u)=N(u)\cap S^k(i)$$，这两个集合的元素个数通常是小于$$k$$的（而如果$$u$$对至少$$k$$个物品评过分的话，$$\mid S^k(i;u)\mid = k$$）。则最终的模型为：

$$
\hat{r_{ui}} = b_{ui} + \mid R^k(i;u)\mid ^{-1/2}\sum_{j\in R(u)} (r_{uj} - b_{uj})\omega_{ij} +\mid N^k(i;u)\mid ^{-1/2}\sum_{j\in N(u)} c_{ij}
$$

使用之前提到的最小化$$f(b_u, b_i, w_{ij}, c_{ij})$$的方法来估计这些参数的取值。记$$e_{ui}=r_{ui} - \hat{r_{ui}}$$，则使用梯度下降法得到的迭代公式如下：

$$
\begin{cases}
b_u \leftarrow b_u+\gamma\cdot (e_{ui} - \lambda_4\cdot b_u) \\
b_i \leftarrow b_i+\gamma\cdot (e_{ui} - \lambda_4\cdot b_i) \\
\omega_{ij} \leftarrow \omega_{ij} + \gamma\cdot(\mid R^k(i;u)\mid ^{-1/2}\cdot e_{ui}\cdot (r_{uj} - b_{uj})-\lambda_4\cdot \omega_{ij}), \forall j \in R^k(i;u) \\
c_{ij} \leftarrow c_{ij} + \gamma\cdot(\mid N^k(i;u)\mid ^{-1/2}\cdot e_{ui}-\lambda_4\cdot c_{ij}), \forall j \in N^k(i;u)
\end{cases}
$$

对于Netflix数据集，Koren推荐取$$\gamma=0.005$$，$$\lambda_4=0.002$$，对所有数据集进行15轮训练。从实际效果来看$$k$$越大，推荐的效果越好。这个模型的计算主要集中在参数训练上，一旦模型训练出来了，就可以快速的进行在线的预测。

### 2.3、推广到基于隐特征的模型

原始的SVD是将用户和物品映射到一个隐特征集合：

$$
\hat{r_{ui}} = b_{ui} + p_u^T\cdot q_i
$$

由于用户的规模通常远大于物品的规模，因此考虑用$$u$$喜欢的物品来对$$u$$进行建模，再加上隐式反馈的数据，可以得到Asymmetric-SVD模型：

$$
\hat{r_{ui}} = b_{ui} + q_i^T(\mid R(u)\mid ^{-1/2}\sum_{j\in R(u)} (r_{uj} - b_{uj})x_j +\mid N(u)\mid ^{-1/2}\sum_{j\in N(u)} y_j)
$$

其中，$$x_j$$和$$y_j$$是用来控制显式反馈和隐式反馈重要性比例的参数。用最小化$$f(b_u,b_i,q_i,x_j,y_j)$$来估计这些参数值。由于这里用$$(r_{uj} - b_{uj})x_j$$来替代原来的用户隐特征，因此数据量少了很多。该模型具有比较好的可解释性，并且对于新用户来讲，只要他做了一些反馈，即更新了$$r_{uj}$$后，就可以立即算出估计值；但是如果新上线一个物品，由于$$q_i^T$$需要重新估计，因此对新物品的冷启动需要一定的反应时间。

如果对于计算不是很care的话，当然可以不用这种简化处理，还是对用户直接进行建模（$$p_u$$），这样的效果会更好一些，但是可解释性之类的就要差一些：

$$
\hat{r_{ui}} = b_{ui} + q_i^T(p_u +\mid N(u)\mid ^{-1/2}\sum_{j\in N(u)} y_j)
$$

### 2.4、联合模型

如果把上面两个模型看成是\`预测值=基准估计+偏移量\`的话，那么这两个模型就可以混合到一起，变成：

$$
\hat{r_{ui}} = b_{ui} + q_i^T(p_u +\mid N(u)\mid ^{-1/2}\sum_{j\in N(u)} y_j) + (\mid R^k(i;u)\mid ^{-1/2}\sum_{j\in R(u)} (r_{uj} - b_{uj})\omega_{ij} +\mid N^k(i;u)\mid ^{-1/2}\sum_{j\in N(u)} c_{ij})
$$

其中，第一项为基准估计，第二项 provides the interaction between the user profile and the item profile. In our example, it may find that “The Sixth Sense” and Joe are rated high on the Psychological Thrillers scale. 第三项 contributes fine grained adjustments that are hard to profile, such as the fact that Joe rated low the related movie “Signs”.

使用梯度下降法得到的迭代公式如下：

$$
\begin{cases}
b_u \leftarrow b_u+\gamma_1\cdot (e_{ui} - \lambda_6\cdot b_u) \\
b_i \leftarrow b_i+\gamma_1\cdot (e_{ui} - \lambda_6\cdot b_i) \\
q_i \leftarrow q_i+ \gamma_2\cdot(e_{ui}\cdot(p_u+\mid N(u)\mid ^{-1/2}\sum_{j\in N(u)} y_j)-\lambda_7\cdot q_i) \\
p_u \leftarrow p_u + \gamma_2\cdot(e_{ui}\cdot q_i - \lambda_7\cdot p_u) \\
y_j \leftarrow y_j+\gamma_2\cdot(e_{ui} \cdot\mid N(u)\mid ^{-1/2} \cdot q_i - \lambda_7\cdot y_j) \\
\omega_{ij} \leftarrow \omega_{ij} + \gamma_3\cdot(\mid R^k(i;u)\mid ^{-1/2}\cdot e_{ui}\cdot (r_{uj} - b_{uj})-\lambda_8\cdot \omega_{ij}), \forall j \in R^k(i;u) \\
c_{ij} \leftarrow c_{ij} + \gamma_3\cdot(\mid N^k(i;u)\mid ^{-1/2}\cdot e_{ui}-\lambda_8\cdot c_{ij}), \forall j \in N^k(i;u)
\end{cases}
$$

在Netflix的数据集上，建议参数为$$\gamma_1=\gamma_2=0.007$$，$$\gamma_3=0.001$$，$$\lambda_6=0.005$$，$$\lambda_7=\lambda_8=0.015$$，整体迭代约30轮收敛，每一轮训练时，可以将$$\gamma_*$$减少10%。而$$k=300$$，再大也不会有明显的性能提升。

最后，Koren还设计了一个比较巧妙的实验，解答了我一直以来一个疑问：RMSE的提升是否也意味着推荐效果的提升。他们设计了一个针对topN推荐的测试，主要的思想是先找出所有5-star的评分，认为这些评分意味着该用户喜欢这部电影，然后对所有这些$$(u,i)$$，随机再选1000部电影，估计$$u$$对这些电影的评分，看用户对这些电影里所有的5-star电影排名情况，然后对不同的算法进行比较，发现RMSE越小的算法，将5-star排到前面的概率也越大，从而说明了在这种情况下，RMSE的提升也意味着推荐效果的提升。

