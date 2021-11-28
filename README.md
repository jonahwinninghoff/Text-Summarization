# Text-Summarization

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jonahwinninghoff/Springboard/graphs/commit-activity)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**[Overview](#overview)** | **[Findings](#findings)** | **[Transformers](#transformers)** | **[Big Bird](#bigbird)** | **[Method](#method)** | **[Dataset](#dataset)** | **[Insights](#insights)** | **[Future](#future)** | **[Conclusion](#concluding)** | **[References](#refer)**

## OVERVIEW <a id='overview'></a>

<p align = 'justify'> For this research, two text summarizations are compared using specific metrics and a timer. Two text summarizations outlined in this research are the Big Bird and XLNET. The set of metrics applied to the comparison is Recall-Oriented Understudy for Gisting Evaluation (ROUGE). The timer with CPU 1.6 GHZ is included to assess the algorithmic efficiency. Both algorithms are Transfer Learnings. Transformers are known for solving various Natural Language Processing (NLP) tasks such as text generation. What leads to this research is a fundamental question to ask. The Google Research team attempts to develop a different approach to address the inherent self-attention mechanism problem in Transformers models called the block sparsity. The research team uses a mathematical assessment to demonstrate the block sparsity that reduces the quadratic dependency to the linear dependency (in relationship of the number of tokens and memory or time) (Zaheer et al., 2020), which is skeptical.</p>

## KEY FINDINGS <a id = 'findings'></a>

<ul>
  <li><p align = 'justify'> As indicated by Randomized Controlled Trial analysis, the Big Bird model performance is significantly higher than XLNET at Bonferroni correction level. </p></li>
  <li><p align = 'justify'>However, XLNET outperforms Big Bird model in memory efficiency based on producing text prediction per article text.</p></li>
  <li><p align = 'justify'>There is evidence that the model produces some redundancies produced by Big Bird text summarization.</p></li>
  <li><p align = 'justify'>Other evidence shows that the Big Bird model reduces quadratic dependency to linear dependency which is against my hypothesis.</p></li>
</ul>

## TRANSFORMERS ARCHITECTURE <a id='transformers'></a>

<p align = 'justify'>To understand what the result of quadratic dependency is, Transformers architecture and its history needs to be addressed first. Before the Transformers architecture is established, the long short-term memory (LSTM) and gated recurrent neural networks are considered as the state-of-the-art approaches in addressing NLP problems. The significant constraint in this model is sequential computation, so the attention mechanisms help to remove the constraint. As a result, the architecture proves to be a milestone in the NLP area throughout the years. </p>

<img src='https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Transformers%20Architecture.png?raw=true'/>(Vaswani et al., 2017)

<p align = 'justify'>As indicated by above, the architecture consists of two multi-layered parts, which are encoder and decoder. Both processings absorb word embeddings processed by word2vec. The difference between encoder and decoder is relatively straightforward. The representation of encoder is X (x<sub>1</sub>, ..., x<sub>n</sub>) while the representation of decoder is Z (z<sub>1</sub>, ..., z<sub>n</sub>). In this case, the representation of encoder is word embedding of unsummarized text. The representation of decoder is word embedding of the actual summarized text.</p>

<p align = 'justify'>The self-attention is associated with <i>head<sub>i</sub></i>, which is associated with <i>multi-head attention</i> as seen by above and below. The self-attention contains a matrix of queries multiplied by transposed key divided by a root square of key dimensionality in softmax function multiplied by values. The softmax is a generalized version of logistic function. The values can be considered as weight that can be updated. This self-attention is built on matrix multiplicaiton code in order to be more space-efficient and faster.</p>

<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Attention-formula.png?raw=true'/>
<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/head_i.png?raw=true'/>
<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/multi-head-attention.png?raw=true'/>

<p align = 'justify'> The <i>head<sub>i</sub></i> has a vector of Q, K, and V each multplied by a vector of weights while the multi-head attention is simply a concatenation of <i>head<sub>i</sub></i> (Vaswani et al., 2017). Based on the graph theory, the problem is that the attention mechansim has quadratic dependency due to the fully connected graph. This is known to be <i>sparsification problem</i> (Zaheer et al., 2020). For this research, the XLNET is used to compare with Big Bird. Even though the change in this model is to use maximum log likelihood of the sequence with respects to permutation, the fundamental of the model remains the same (Yang et al., 2020). </p>

## BIG BIRD <a id ='bigbird'></a>

<p align = 'justify'>As mentioned earlier, this architecture has a sparsification problem where the normal attention has a full connection leading to the quadratic increase in memory or time term. The Google Research team attempts to remedy the problem using the block sparsity. In other words, the block sparsity consists of three different types of connections called global, sliding, and random. For example, if the sentence states, "How you have been so far?", this sentence has six tokens that have particular ways to connect based on three types.</p>

<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Block%20Sparse.png?raw=true'/>

<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Compare%20attentions.png?raw=true'/>

<p align = 'justify'>However, the concept of global and sliding connections is not novel, which is similar to the article <i>Generating Long Sequences with Sparse Transformers</i> (Child et al., 2019). What makes the Big Bird algorithm different is random connection. The number of connections in block sparsity is less than the connections in normal attention. This reduction may be smaller but it becomes more significant when the length of sequence is increased.</p>

<p align = 'justify'>Using the random connection is a concern due to the lack of algorithmic intentions. Having intention in the algorithm is important in order to ensure that the model performs well. On another hand, the Google Research team seems to develop the algorithm based on Central Limit Theorem (CLT) and Law of Large Number (LLN). In other words, their assumption is that the predicted summary becomes consistent when being converged based on the length of sequence. There is an alternative suggestion that may remedy this type of connection, which will be discussed after the model assessment completion.</p>

<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/bird%20view.png?raw=true'/>(Gupta, 2021)

<p align = 'justify'>As seen by above, each color is associated with global, sliding, and random connections while each white slot represents no connection. For example, there is no connection between "work" and "is." This approach reduces time complexity, which comes with the price of no theoretical guarantees as the Google Research acknowledges (Zaheer et al., 2020).</p>

## METHOD <a id ='method'></a> 

<p align = 'justify'>In order to experiment on both text summarizations, there are two approaches that will be used: partial NLP data science and Randomized Controlled Trials (RCTs). The partial NLP data science is a life cycle pipeline that only includes literature review, data quality assessment, data cleanup, exploratory analysis, and feature engineering. This cycle does not include predictive modeling. As mentioned earlier, Big Bird and XLNET both are Transfer Learnings. The next part is RCTs, which randomly sample the ArXiv journals to summarize using these pre-trained models and undertake statistical inferences. This analysis contains features and target variables, such as the following:</p>

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

<p align = 'justify'>The ROUGE-N F1-score is a measure of model accuracy based on the number of matching n-grams between predicted summary and ground-truth summary. For example, ROUGE-1 measures the number of matching unigram while the ROUGE-2 measures bigram. But the ROUGE-L is slightly different. This metric measures the longest common subsequence (LCS) between predicted summary and ground-truth summary. The LCS refers to the maximum length of tokens in total. Data collection on both models takes two days to compute.</p>

## DATASET <a id ='dataset'></a>

<p align = 'justify'>As mentioned earlier, the ArXiv journals are in use to infer models prepared by TensorFlow. This dataset contains three features, which are <i>article id</i>, <i>article text</i>, and <i>actual abstract text</i>. There are three subsets in this dataset, which are testing (6,658 entities), training (119,924 entities), and validation (6,633 entities) sets. For this research, the validation set is in use to evaluate on both models. Based on the exploratory analysis, 70.8% of tokens in article texts matches NLTK dictionaries while 62.05% in abstract text matches these dictionaries. The Big Bird model is pre-trained with Wikipedia dataset (Zaheer et al., 2020) while XLNET model is pre-trained with several datasets other than ArXiv dataset (Yang et al., 2020), so the validation set is considered as an unseen dataset. However, using the entire set is infeasible and time-consuming. For this reason, the ArXiv journals are randomly sampled. The sampling size is 110 for each model.</p>

## ACTIONABLE INSIGHTS <a id ='insights'></a>

<p align = 'justify'>After the data is collected, the information is assessed with statistical inference and descriptive statistics. Before the actionable insights are discussed, one part is important to mention. Being analyzing the models with three different metrics, the Bonferroni correction is first applied. The correction is in use to prevent Type I error albeit being conservative. This correction is prone to have the Type II error (the null hypothesis is failed to reject when false), which can be concerning in general. On another hand, this research is focused on Big Bird's model performance comparing to XLENT. Big Bird model does outperform XLNET at the significance level in every metrics. This research attempts to address three different questions in order, as the following:</p>

<ul>
  <li><p align = 'justify'> Does the Big Bird model outperform XLNET model on predicted summary term?</p></li>
  <li><p align = 'justify'> Being compared with XLNET model, what is speed of each predicted summary that Big Bird produce?</p></li>
  <li><p align = 'justify'> Does the Big Bird successfully reduce this quadratic dependency to linear dependency in sequence term?</p></li>
</ul>
  
<div align = 'center'><img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/rouges.png?raw=true'/></div>

<p align = 'justify'> As indicated by above, the black color (or “0”) represents the XLNET model while the second color (or “1”) represents the Big Bird model. The confidence interval is 95% with Bonferroni correction, which is wider than without this correction. Average ROUGE-1 for Big Bird and XLNET is 57.65% with 4.66% margin of error and 25.66% with 2.27% margin of error, respectively, while average ROUGE-2 is 48.64% with 5.67% margin of error and 5.57% with 0.93% margin of error. Average ROUGE-L is 52.78% with 5.02% margin of error and 14.46% with 1.01% margin of error. In short, the Big Bird model does outperform the XLNET model at a significance level. However, using this model to predict the summary with CPU 1.6 GHZ is 25.8 minutes by median (26.6 minutes by mean). In other words, this model processor is slightly slower than the average reading speed. The average reading speed is 200 words per minute (Rayner et al., 2010). In other words, the model is not well-scalable with local application.</p>

<div align = 'center'><img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/timer.png?raw=true'/></div>

<p align = 'justify'>There is no evidence of time overlap between Big Bird and XLNET models. For that reason, the violin plot is in use. As indicated by above, XLNET is much faster than Big Bird. Both distributions are leptokurtic and right-skewed on memory term or time. However, Big Bird model has a higher right-skewed and kurtotic score than XLNET model. That is the reason why skepticism exists on Big Bird algorithmic efficiency. When the loess is applied, the result is surprising. Before the discussion goes further, the reason for using loess needs to be explained. The advantage of the loess tool is non-parametric, so the tool helps to focus on the relationship between time and word counts with minimal assumptions.
</p>

<div align = 'center'><img src = "https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/comparing.gif?raw=true"/></div>

<p align = 'justify'>There are two sliders with two different plots. The first slider is time per predicted summary while the second slider is with logarithm. To mitigate any outlier problems, the logarithm is in use. The outliers may have a significant impact on loess regression. Both sliders make same confirmation that Big Bird algorithm successfully establish a linear relationship between the number of tokens and time per predicted summary, which is surprising. The Big Bird algorithm turns out to be successful in text summarization area.</p>


## FUTURE RESEARCH <a id = 'future'></a>

""

## CONCLUSION <a id='concluding'></a>


## REFERENCES <a id = 'refer'></a>

https://papers.nips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf
https://arxiv.org/pdf/1904.10509.pdf
https://arxiv.org/pdf/1906.08237v2.pdf
https://pubmed.ncbi.nlm.nih.gov/21169577/



