\# 论文阅读——Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model



标签（空格分隔）： svd++ paper koren



---



本文对协同过滤中最主要的两种方法（基于邻域的方法和基于隐特征模型的方法）分别提出了优化方案，并且设计了一个联合模型将两种方法统一，从而达到更好的效果。为了进行区分，本文将对SVD进行优化的方案称为SVD+，将联合模型的方法称为SVD++。



\# 研究背景



Koren在做Netflix的比赛过程中，发现基于邻域的方法和基于隐特征模型的方法各有所长：



\|比较\|基于邻域的方法\|基于隐特征模型的方法\|

\|---\|---\|---\|

\|主要思想\|核心在于计算用户/物品的相似度，将相似用户的喜好推荐给用户，或将用户喜欢物品的相似仿物品推荐给用户\|假设真正描述用户评分矩阵性质的内存特征（可能未知）其实只有少数几个，将用户和物品都映射到这些隐特征层，从而使得用户和物品直接关联起来\|

\|挖掘信息特征\|能够对局部强相关的关系更敏感，而无法捕捉全局弱相关的关系\|能够估计关联所有物品/用户的整体结构，但是难以反映局部强相关的关系\|



因此，这两种方法存在天然的互补关系。另外，Koren还发现，使用隐式反馈的数据能够提高推荐的准确性，而这两种方法都不支持使用隐式反馈的数据。基于这些发现，Koren先分别将隐式反馈集成到两个模型中去，得到两个优化的模型，再提出一种联合模型，将这两个优化的模型进一步融合，从而得到更好的效果。



\# 模型推导



文章从Baseline的模型，通过加入各种考虑因素，推导出基于邻域和基于隐特征的两个模型，再推导出联合模型。



\#\# 1、Baseline模型



Baseline模型就是基于历史数据的简单统计，主要看用户$u$的平均评分$b\_u$、电影$i$的平均评分$b\_i$和所有电影的平均评分$\mu$：

$$b\_{ui} = \mu + b\_u + b\_i$$

所有后面的模型都是对这个基准模型的修正。这个基准模型中的参数都是可以离线计算的，用的方法也是本文通用的参数估计方法，先定义损失函数$l\(P\)$：



$$l\(p\_1,p\_2,\cdots\) = \sum\_{\(u,i\)\in \kappa} \(r\_{ui} - \hat{r\_{ui}}\)^2 + \lambda\(\sum\_{p\_1} p\_1^2 + \sum\_{p\_2} p\_2^2 + \cdots\)$$



其中，$P=\{p\_1,p\_2,\cdots\}$表示待估计的参数，$\kappa$表示所有显式反馈的组合（即用户$u$对物品$i$进行过评分），$r\_{ui}$表示评分的实际值，$\hat{r\_{ui}}$表示评分的预测值，$\lambda$为超参，根据经验设置，然后求最小化$l\(P\)$下各参数的值，通常使用最小二乘法，或者文中使用的梯度下降法（效率更高）。比如这个地方，参数就$b\_u$和$b\_i$，可以根据下式进行参数估计：



$$\min\_{b\_\*}\sum\_{\(u,i\)\in \kappa} \(r\_{ui} - \hat{r\_{ui}}\)^2 + \lambda\(\sum\_{p\_1} p\_1^2 + \sum\_{p\_2} p\_2^2 + \cdots\)$$



\#\# 2、推广到基于邻域的模型

本文主要考虑ItemCF，对于两个物品$i$和$j$，它们的相似性$s\_{ij}$是基于Pearson相关系数$\rho\_{ij}$计算得到：



$$s\_{ij} = \frac{n\_{ij}}{n\_{ij}+\lambda\_2}\rho\_{ij}, \ \ \rho\_{ij}=\frac{E\(\(x-\mu\_x\)\(y-\mu\_y\)\)}{\sigma\_x\sigma\_y}$$



其中，$n\_{ij}$表示同时对$i$和$j$进行评分的用户数，$\lambda\_2$应该是防止$i$和$j$比较冷门的情况下，恰好有个别用户同时对它们进行了评分，这时候它们的相关性实际是看不出来的，属于偶然情况，通常$\lambda\_2=100$。之前的ItemCF进一步利用用户$u$评过分的与$i$最相关的$k$个物品$S^k\(i;u\)$来估计用户$u$对$i$的评分：



$$\hat{r\_{ui}} = b\_{ui} + \frac{\sum\_{j\in S^k\(i;u\)} s\_{ij}\(r\_{uj} - b\_{uj}\)}{\sum\_{j\in S^k\(i;u\)} s\_{ij}}$$



但是如果$u$没有对与$i$相似的物品评过分，那上式就主要取决于$b\_{ui}$了。为了解决这个小问题，有方案先计算插值权重$\theta\_{ij}^u$来取代实际的评分：



$$\hat{r\_{ui}} = b\_{ui} + \sum\_{j\in S^k\(i;u\)} \theta\_{ij}^u \(r\_{uj} - b\_{uj}\)$$



但是以上模型都只考虑了用户$u$，而对全局结构没有一个很好的理解，因此Koren提出不仅仅使用用户$u$的对$i$最相关的$k$个物品的评分数据，而是使用所有$u$的评分数据，因此引入一个参数$\omega\_{ij}$来表示$j$的评分对$i$评分的影响，并且这个$\omega\_{ij}$是基于所有用户对$i$和$j$评分估计出来的：



$$\hat{r\_{ui}} = b\_{ui} + \sum\_{j\in R\(u\)} \(r\_{uj} - b\_{uj}\)\omega\_{ij}$$



分析这个式子，当$i$和$j$越相关，说明$j$对$i$的影响越大，即$w\_{ij}$越大，这时候如果$\(r\_{uj} - b\_{uj}\)$较大，则估计的评分相对于$b\_{ui}$的偏移也就越多；反之，当$w\_{ij}$较小时，无论$j$的评分如何都对偏移影响不大。



在此基础上，进一步引入隐式反馈的数据：



$$\hat{r\_{ui}} = b\_{ui} + \sum\_{j\in R\(u\)} \(r\_{uj} - b\_{uj}\)\omega\_{ij} +\sum\_{j\in N\(u\)} c\_{ij}$$



其中，$c\_{ij}$表示隐式反馈对基准估计的偏移影响，当$j$与$i$的评分强相关时，$c\_{ij}$较大。这个式子的主要问题是，它对重度用户的推荐和对轻度用户的推荐结果相差较大，因为重度用户的显式反馈和隐式反馈都很多，因此偏移项值较大。Koren发现，做一下规范化以后，效果会更好：



$$\hat{r\_{ui}} = b\_{ui} + \mid R\(u\)\mid ^{-1/2}\sum\_{j\in R\(u\)} \(r\_{uj} - b\_{uj}\)\omega\_{ij} +\mid N\(u\)\mid ^{-1/2}\sum\_{j\in N\(u\)} c\_{ij}$$



为了降低上式的计算复杂度，可以只考虑对$i$影响最大的$k$个物品，记$R^k\(i;u\)=R\(u\)\cap S^k\(i\)$表示$u$评分过的物品中属于$i$最相似的Top k物品，类似的，记$N^k\(i;u\)=N\(u\)\cap S^k\(i\)$，这两个集合的元素个数通常是小于$k$的（而如果$u$对至少$k$个物品评过分的话，$\mid S^k\(i;u\)\mid = k$）。则最终的模型为：



$$\hat{r\_{ui}} = b\_{ui} + \mid R^k\(i;u\)\mid ^{-1/2}\sum\_{j\in R\(u\)} \(r\_{uj} - b\_{uj}\)\omega\_{ij} +\mid N^k\(i;u\)\mid ^{-1/2}\sum\_{j\in N\(u\)} c\_{ij}$$



使用之前提到的最小化$f\(b\_u, b\_i, w\_{ij}, c\_{ij}\)$的方法来估计这些参数的取值。记$e\_{ui}=r\_{ui} - \hat{r\_{ui}}$，则使用梯度下降法得到的迭代公式如下：

$$\begin{cases}

b\_u \leftarrow b\_u+\gamma\cdot \(e\_{ui} - \lambda\_4\cdot b\_u\) \\

b\_i \leftarrow b\_i+\gamma\cdot \(e\_{ui} - \lambda\_4\cdot b\_i\) \\

\omega\_{ij} \leftarrow \omega\_{ij} + \gamma\cdot\(\mid R^k\(i;u\)\mid ^{-1/2}\cdot e\_{ui}\cdot \(r\_{uj} - b\_{uj}\)-\lambda\_4\cdot \omega\_{ij}\), \forall j \in R^k\(i;u\) \\

c\_{ij} \leftarrow c\_{ij} + \gamma\cdot\(\mid N^k\(i;u\)\mid ^{-1/2}\cdot e\_{ui}-\lambda\_4\cdot c\_{ij}\), \forall j \in N^k\(i;u\)

\end{cases}$$



对于Netflix数据集，Koren推荐取$\gamma=0.005$，$\lambda\_4=0.002$，对所有数据集进行15轮训练。从实际效果来看$k$越大，推荐的效果越好。这个模型的计算主要集中在参数训练上，一旦模型训练出来了，就可以快速的进行在线的预测。



\#\# 3、推广到基于隐特征的模型



原始的SVD是将用户和物品映射到一个隐特征集合：



$$\hat{r\_{ui}} = b\_{ui} + p\_u^T\cdot q\_i$$



由于用户的规模通常远大于物品的规模，因此考虑用$u$喜欢的物品来对$u$进行建模，再加上隐式反馈的数据，可以得到Asymmetric-SVD模型：



$$\hat{r\_{ui}} = b\_{ui} + q\_i^T\(\mid R\(u\)\mid ^{-1/2}\sum\_{j\in R\(u\)} \(r\_{uj} - b\_{uj}\)x\_j +\mid N\(u\)\mid ^{-1/2}\sum\_{j\in N\(u\)} y\_j\)$$



其中，$x\_j$和$y\_j$是用来控制显式反馈和隐式反馈重要性比例的参数。用最小化$f\(b\_u,b\_i,q\_i,x\_j,y\_j\)$来估计这些参数值。由于这里用$\(r\_{uj} - b\_{uj}\)x\_j$来替代原来的用户隐特征，因此数据量少了很多。该模型具有比较好的可解释性，并且对于新用户来讲，只要他做了一些反馈，即更新了$r\_{uj}$后，就可以立即算出估计值；但是如果新上线一个物品，由于$q\_i^T$需要重新估计，因此对新物品的冷启动需要一定的反应时间。



如果对于计算不是很care的话，当然可以不用这种简化处理，还是对用户直接进行建模（$p\_u$），这样的效果会更好一些，但是可解释性之类的就要差一些：



$$\hat{r\_{ui}} = b\_{ui} + q\_i^T\(p\_u +\mid N\(u\)\mid ^{-1/2}\sum\_{j\in N\(u\)} y\_j\)$$



\#\# 4、联合模型

如果把上面两个模型看成是\`预测值=基准估计+偏移量\`的话，那么这两个模型就可以混合到一起，变成：



$$\hat{r\_{ui}} = b\_{ui} + q\_i^T\(p\_u +\mid N\(u\)\mid ^{-1/2}\sum\_{j\in N\(u\)} y\_j\) + \(\mid R^k\(i;u\)\mid ^{-1/2}\sum\_{j\in R\(u\)} \(r\_{uj} - b\_{uj}\)\omega\_{ij} +\mid N^k\(i;u\)\mid ^{-1/2}\sum\_{j\in N\(u\)} c\_{ij}\)$$



其中，第一项为基准估计，第二项 provides the interaction between the user profile and the item profile. In our example, it may find that “The Sixth Sense” and Joe are rated high on the Psychological Thrillers scale. 第三项 contributes fine grained adjustments that are hard to profile, such as the fact that Joe rated low the related movie “Signs”.



使用梯度下降法得到的迭代公式如下：

$$\begin{cases}

b\_u \leftarrow b\_u+\gamma\_1\cdot \(e\_{ui} - \lambda\_6\cdot b\_u\) \\

b\_i \leftarrow b\_i+\gamma\_1\cdot \(e\_{ui} - \lambda\_6\cdot b\_i\) \\

q\_i \leftarrow q\_i+ \gamma\_2\cdot\(e\_{ui}\cdot\(p\_u+\mid N\(u\)\mid ^{-1/2}\sum\_{j\in N\(u\)} y\_j\)-\lambda\_7\cdot q\_i\) \\

p\_u \leftarrow p\_u + \gamma\_2\cdot\(e\_{ui}\cdot q\_i - \lambda\_7\cdot p\_u\) \\

y\_j \leftarrow y\_j+\gamma\_2\cdot\(e\_{ui} \cdot\mid N\(u\)\mid ^{-1/2} \cdot q\_i - \lambda\_7\cdot y\_j\) \\

\omega\_{ij} \leftarrow \omega\_{ij} + \gamma\_3\cdot\(\mid R^k\(i;u\)\mid ^{-1/2}\cdot e\_{ui}\cdot \(r\_{uj} - b\_{uj}\)-\lambda\_8\cdot \omega\_{ij}\),\ \forall j \in R^k\(i;u\) \\

c\_{ij} \leftarrow c\_{ij} + \gamma\_3\cdot\(\mid N^k\(i;u\)\mid ^{-1/2}\cdot e\_{ui}-\lambda\_8\cdot c\_{ij}\),\ \forall j \in N^k\(i;u\)

\end{cases}$$



在Netflix的数据集上，建议参数为$\gamma\_1=\gamma\_2=0.007$，$\gamma\_3=0.001$，$\lambda\_6=0.005$，$\lambda\_7=\lambda\_8=0.015$，整体迭代约30轮收敛，每一轮训练时，可以将$\gamma\_\*$减少10%。而$k=300$，再大也不会有明显的性能提升。



最后，Koren还设计了一个比较巧妙的实验，解答了我一直以来一个疑问：RMSE的提升是否也意味着推荐效果的提升。他们设计了一个针对topN推荐的测试，主要的思想是先找出所有5-star的评分，认为这些评分意味着该用户喜欢这部电影，然后对所有这些$\(u,i\)$，随机再选1000部电影，估计$u$对这些电影的评分，看用户对这些电影里所有的5-star电影排名情况，然后对不同的算法进行比较，发现RMSE越小的算法，将5-star排到前面的概率也越大，从而说明了在这种情况下，RMSE的提升也意味着推荐效果的提升。

