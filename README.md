# Comparing Performance and Accuracy of Big Bird and XLNet for Text Summarization

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)](https://github.com/jonahwinninghoff/Springboard/graphs/commit-activity)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

See the presentation ([PPT](https://github.com/jonahwinninghoff/Text-Summarization/raw/main/Presentation/BigBird.pptx) or [PDF](https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Presentation/BigBird.pdf))

**[Overview](#overview)** | **[Findings](#findings)** | **[Transformers](#transformers)** | **[Big Bird](#bigbird)** | **[Method](#method)** | **[Dataset](#dataset)** | **[Insights](#insights)** | **[Future](#future)** | **[Conclusion](#concluding)** | **[References](#refer)**

## OVERVIEW <a id='overview'></a>

<p align = 'justify'> For this research, two text summarizations are compared using specific metrics and a timer. Two text summarizations outlined in this research are the Big Bird and XLNet. The set of metrics applied to the comparison is Recall-Oriented Understudy for Gisting Evaluation (ROUGE). The timer with CPU 1.6 GHZ is included to assess the algorithmic efficiency. Both algorithms are Transfer Learnings. Transformers are known for solving various Natural Language Processing (NLP) tasks such as text generation. What leads to this research is a fundamental question to ask. The Google Research team attempts to develop a different approach to address the inherent self-attention mechanism problem in Transformers models called block sparsity. The research team uses a mathematical assessment to demonstrate the block sparsity that reduces the quadratic dependency to the linear dependency (in a relationship of the number of tokens and memory or time) (Zaheer et al., 2020), which is skeptical.</p>

## KEY FINDINGS <a id = 'findings'></a>

<ul>
  <li><p align = 'justify'> As indicated by Randomized Controlled Trial analysis, the Big Bird model performance is significantly higher than XLNet at the Bonferroni correction level. </p></li>
  <li><p align = 'justify'>However, XLNet outperforms the Big Bird model in memory efficiency based on producing text prediction per article text.</p></li>
  <li><p align = 'justify'>There is evidence that the model produces some redundancies produced by Big Bird text summarization.</p></li>
  <li><p align = 'justify'>Other evidence shows that the Big Bird model reduces quadratic dependency to linear dependency, which is against my hypothesis.</p></li>
</ul>

## TRANSFORMERS ARCHITECTURE <a id='transformers'></a>

<p align = 'justify'>To understand what the result of quadratic dependency is, the Transformer’s architecture and its history need to be addressed first. Before the Transformers architecture is established, the long short-term memory (LSTM) and gated recurrent neural networks are considered the state-of-the-art approaches in addressing NLP problems. The significant constraint in this model is sequential computation, so the attention mechanisms help to remove the constraint. As a result, the architecture proves to be a milestone in the NLP area throughout the years. </p>

<img src='https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Transformers%20Architecture.png?raw=true'/>(Vaswani et al., 2017)

<p align = 'justify'>As indicated above, the architecture consists of two multi-layered parts, which are the encoder and decoder. Both processings absorb word embeddings processed by word2vec. The difference between an encoder and a decoder is relatively straightforward. The representation of encoder is X (x<sub>1</sub>, ..., x<sub>n</sub>) while the representation of decoder is Z (z<sub>1</sub>, ..., z<sub>n</sub>). In this case, the representation of the encoder is word embedding of unsummarized text. The representation of the decoder is word embedding of the actual summarized text.</p>

<p align = 'justify'>The self-attention is associated with <i>head<sub>i</sub></i>, which is associated with <i>multi-head attention</i> as seen by above and below. The self-attention contains a matrix of queries multiplied by the transposed key divided by a root square of key dimensionality in the softmax function multiplied by values. The softmax is a generalized version of the logistic function. The values can be considered as weights that can be updated. This self-attention is built on matrix multiplication code in order to be more space-efficient and faster.</p>

<p align = 'justify'> The <i>head<sub>i</sub></i> has a vector of Q, K, and V, each multiplied by a vector of weights, while the multi-head attention is simply a concatenation of <i>head<sub>i</sub></i> (Vaswani et al., 2017). Based on graph theory, the problem is that the attention mechanism has quadratic dependency due to its being a fully connected graph. This is known to be <i>sparsification problem</i> (Zaheer et al., 2020). For this research, the XLNet is used to compare with Big Bird. The change in this model is to use the maximum log-likelihood of the sequence with respect to permutation (Yang et al., 2020). </p>

## BIG BIRD <a id ='bigbird'></a>

<p align = 'justify'>As mentioned earlier, this architecture has a sparsification problem where the normal attention has a full connection leading to the quadratic increase in memory or time term. The Google Research team attempts to remedy the problem using the block sparsity. In other words, the block sparsity consists of three different types of connections: global, sliding, and random. For example, if the sentence states, "How you have been so far?", this sentence has six tokens that have particular ways to connect based on three types.</p>

<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Block%20Sparse.png?raw=true'/>

<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Compare%20attentions.png?raw=true'/>

<p align = 'justify'>However, the concept of global and sliding connections is not novel, which is similar to the article <i>Generating Long Sequences with Sparse Transformers</i> (Child et al., 2019). What makes the Big Bird algorithm different is the random connection. The number of connections in block sparsity is less than the connections in normal attention. This reduction may be smaller, but it becomes more significant when the length of the sequence is increased.</p>

<p align = 'justify'>Using the random connection is a concern due to the lack of algorithmic intentions. Having intention in the algorithm is important in order to ensure that the model performs well. On the other hand, the Google Research team seems to have developed the algorithm based on the Central Limit Theorem (CLT) and the Law of Large Numbers (LLN). In other words, their assumption is that the predicted summary becomes consistent when being converged based on the length of the sequence. There is an alternative suggestion that may remedy this type of connection, which will be discussed after the model assessment completion.</p>

<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/bird%20view.png?raw=true'/>(Gupta, 2021)

<p align = 'justify'>As seen above, each color is associated with global, sliding, and random connections, while each white slot represents no connection. For example, there is no connection between "work" and "is." This approach reduces time complexity, which comes with the price of no theoretical guarantees, as Google Research acknowledges (Zaheer et al., 2020).</p>

## METHOD <a id ='method'></a> 

<p align = 'justify'>In order to experiment on both text summarizations, there are two approaches that will be used: partial NLP data science and Randomized Controlled Trials (RCTs). The partial NLP data science is a life cycle pipeline that only includes a literature review, data quality assessment, data cleanup, exploratory analysis, and feature engineering. This cycle does not include predictive modeling. As mentioned earlier, Big Bird and XLNet are Transfer Learnings. The next part is RCTs, which randomly sample the ArXiv journals to summarize using these pre-trained models and undertake statistical inferences. This analysis contains features and target variables, such as the following:</p>

Features:

<ul>
  <li><i>article id</i></li>
  <li><i>article text</i></li>
  <li><i>actual abstract text</i></li>
  <li><i>predicted summary</i></li>
  <li><i>article word counts</i></li>
  <li><i>abstract word counts</i></li>
  <li><i>predicted summary word counts</i></li>
  <li><i>big bird (binary category)</i></li>
</ul>

Target variables:
<ul>
  <li><i>time per predicted summary (in seconds)</i></li>
  <li><i>rouge 1 F1 score</i></li>
  <li><i>rouge 2 F1 score</i></li>
  <li><i>rouge L F1 score</i></li>
</ul>

<p align = 'justify'>The ROUGE-N F1-score is a measure of model accuracy based on the number of matching n-grams between the predicted summary and ground-truth summary. For example, ROUGE-1 measures the number of matching unigrams while ROUGE-2 measures bigrams. But the ROUGE-L is slightly different. This metric measures the longest common subsequence (LCS) between the predicted summary and ground-truth summary. The LCS refers to the maximum length of tokens in total. Data collection on both models takes two days to compute.</p>

## DATASET <a id ='dataset'></a>

<p align = 'justify'>As mentioned earlier, the ArXiv journals are used to infer models prepared by TensorFlow. This dataset contains three features, which are <i>article id</i>, <i>article text</i>, and <i>actual abstract text</i>. There are three subsets in this dataset: testing (6,658 entities), training (119,924 entities), and validation (6,633 entities). For this research, the validation set is used to evaluate both models. Based on the exploratory analysis, 70.8% of tokens in article texts match NLTK dictionaries, while 62.05% in the abstract text match these dictionaries. The Big Bird model is pre-trained with the Wikipedia dataset (Zaheer et al., 2020), while the XLNet model is pre-trained with several datasets other than the ArXiv dataset (Yang et al., 2020), so the validation set is considered as an unseen dataset. However, using the entire set is infeasible and time-consuming. For this reason, the ArXiv journals are randomly sampled. The sampling size is 110 for each model.</p>

## ACTIONABLE INSIGHTS <a id ='insights'></a>

<p align = 'justify'>After the data is collected, the information is assessed with statistical inference and descriptive statistics. Before the actionable insights are discussed, one part is important to mention. After analyzing the models with three different metrics, the Bonferroni correction is first applied. The correction is in use to prevent Type I error, albeit being conservative. This correction is prone to Type II error (the null hypothesis fails to reject when false), which can be concerning in general. On the other hand, this research is focused on Big Bird's model performance compared to XLENT. Big Bird model does outperform XLNet at the significance level in every metric listed in this research. This research attempts to address three different questions in order as the following:</p>

<ul>
  <li><p align = 'justify'> Does the Big Bird model outperform the XLNet model on the predicted summary term?</p></li>
  <li><p align = 'justify'> Being compared with the XLNet model, how fast can Big Bird produce each predicted summary?</p></li>
  <li><p align = 'justify'> Does the Big Bird successfully reduce this quadratic dependency to linear dependency in sequence terms?</p></li>
</ul>
  
<div align = 'center'><img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/rouges.png?raw=true'/></div>

<p align = 'justify'> As indicated above, the black color (or “0”) represents the XLNet model, while the second color (or “1”) represents the Big Bird model. The confidence interval is 95% with the Bonferroni correction, which is wider than without this correction. Average ROUGE-1 for Big Bird and XLNet is 57.65% with a 4.66% margin of error and 25.66% with a 2.27% margin of error, respectively, while average ROUGE-2 is 48.64% with a 5.67% margin of error and 5.57% with 0.93% margin of error. Average ROUGE-L is 52.78% with a 5.02% margin of error and 14.46% with a 1.01% margin of error. In short, the Big Bird model does outperform the XLNet model at a significance level. However, using this model to predict the summary with CPU 1.6 GHZ is 25.8 minutes by median (26.6 minutes by mean). In other words, this model processor is slightly faster than the average reading speed. The average reading speed is 200 words per minute (Rayner et al., 2010). In other words, the model is not well-scalable with local applications.</p>

<div align = 'center'><img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/timer.png?raw=true'/></div>

<p align = 'justify'>There is no evidence of time overlap between Big Bird and XLNet models. For that reason, the violin plot is used. As indicated above, XLNet is much faster than Big Bird. Both distributions are leptokurtic and right-skewed on memory term or time. However, the Big Bird model has a higher right-skewed and kurtotic score than the XLNet model. That is the reason why skepticism exists about Big Bird's algorithmic efficiency. When the loess is applied, the result is surprising. Before the discussion goes further, the reason for using loess needs to be explained. The advantage of the loess tool is that it is non-parametric, so the tool helps to focus on the relationship between time and word counts with minimal assumptions.
</p>

<div align = 'center'><img src = "https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/comparing.gif?raw=true"/></div>

<p align = 'justify'>There are two sliders with two different plots. The first slider is time per predicted summary, while the second slider is with logarithm. Logarithm is used to mitigate any outlier problems. The outliers may have a significant impact on loess regression. Both sliders confirm that the Big Bird algorithm successfully establishes a linear relationship between the number of tokens and time per predicted summary, which is surprising. Besides this scalability issue, the Big Bird algorithm turns out to be successful in the text summarization area.</p>


## FUTURE RESEARCH <a id = 'future'></a>

<p align = 'justify'>In the Big Bird algorithm, there are two issues that have been identified. The first problem is, unsurprisingly, scalability, and the second problem is random connection in block sparse attention. Both problems can be, for future research, a challenge. To make this algorithm scalable, each self-attention layer in the original architecture needs to be replaced with an Attention Free Transformer (AFT). Not only that, but the algorithm also needs to modify block sparse attention. In doing so, the Transformer with AFT needs to be assessed based on model performance and scalability. Then, it needs to be compared with the Big Bird algorithm. The next experiment is to replace each self-attention layer with AFT while persevering block sparse attention. Another experiment is to modify block sparse attention without self-attention replacement. The final experiment is self-attention replacement with modified block sparse attention. In other words, there are five different experiments in total to determine which algorithm outperforms others.</p>

<p align = 'justify'>The modified block sparse attention is to replace random connection with Bayesian connection inspired by Bayesian optimization. This optimization consists of three functions: objective, acquisition, and surrogate. The objective function possesses a true shape that is unobservable and can only reveal some data points, which can otherwise be expensive to compute, while the surrogate function is the probabilistic model being built to exploit what is known, and it can be updated based on new information. The acquisition function is to calculate, in this case, the adjacency matrix that is likely to yield the higher local maximum of the objective function using a surrogate function (Brochu et al., 2010).</p> 

<p align = 'justify'>From now on, future research will focus on AFT due to its advantages. In the Computer Vision area, a recent study shows that the AFT proves to be efficient with high-yield results (Zhai et al., 2021). However, the AFT for text summarization has not been tested in this paper. This is the reason why, in the future, scalability and block sparsity need to be evaluated with both relatively novel approaches.</p>


## CONCLUSION <a id='concluding'></a>

<p align = 'justify'>Both Big Bird and XLNet models are tested for performance and efficiency using local applications. There is a clear trade-off between accuracy and efficiency with both algorithms. For example, the Big Bird model does better with predicting summary. This algorithm successfully linearizes the self-attention mechanism using block sparsity. Using the cloud environment with the Big Bird model is a prerequisite for efficiency. On the other hand, the Big Bird algorithm is highly recommended for producing summaries as long the cloud environment is used. This model has scalability and redundant problems, as seen in several predicted texts. For future research, in order to determine if the novel algorithm can be improved, the AFT and Bayesian connection are strongly recommended to be tested.</p>

## REFERENCES <a id = 'refer'></a>

<p>Brochu, E., Cora, VM., and Freitas, N. “A Tutorial on Bayesian Optimization of Expensive Cost Functions, With Application to Active User Modeling and Hierarchical Reinforcement Learning.” arXiv, Dec. 2020.
https://www.math.umd.edu/~slud/RITF17/Tutorial_on_Bayesian_Optimization.pdf</p>

<p>Child, R., Gray, S., Radford, A., and Sutskever, I. “Generating Long Sequences with Sparse Transformers.” arXiv, 2019. https://arxiv.org/pdf/1904.10509.pdf</p>

<p>Rayner, K., Slattery, TJ., and Bélanger, NN. “Eye movements, the perceptual span, and reading speed.” Psychon Bull Rev., Dec. 2010. doi: 10.3758/PBR.17.6.834</p>

<p>Gupta, V. “Understanding BigBird’s Block Sparse Attention.” Huggingface, Mar. 2021. https://huggingface.co/blog/big-bird</p>

<p>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, Ł., Gomez, AN., Kaiser, L., and Polosukhin, I. “Attention is All You Need.” Advances in Neural Information Processing Systems 30. NIPS, 2017.
https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf</p>

<p>Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R., and Le., QV. “XLNet: Generalized Autoregressive Pretraining for Language Understanding.” arXiv, Jan. 2020.https://arxiv.org/pdf/1906.08237v2.pdf</p>

<p>Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S., Philip, P., Ravula, A., Wang, Q., Yang, L., and Amr Ahmed, A. “Big Bird: Transformers for Longer Sequences.” Advances in Neural Information Processing Systems 33, NeurIPS, 2020. https://papers.nips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf</p>

<p>Zhai, S., Talbott, W., Srivastava, N., Huang, C., Goh, H., Zhang, R., and Susskind, J. “An Attention Free Transformer.” arXiv, Sep. 2021. https://arxiv.org/pdf/2105.14103.pdf</p>
