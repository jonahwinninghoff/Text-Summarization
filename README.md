# Text-Summarization

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jonahwinninghoff/Springboard/graphs/commit-activity)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**[Overview](#overview)** | **[Findings](#findings)** | **[Transformers](#transformers)** | **[Big Bird](#bigbird)** | **[Method](#method)** | **[Dataset](#dataset)** | **[Insights](#insights)** | **[Future](#future)** | **[Conclusion](#concluding)** | **[References](#refer)**

## OVERVIEW <a id='overview'></a>

<p align = 'justify'> For this research, two text summarizations are compared using specific metrics and a timer. Two text summarizations outlined in this research are the Big Bird and XLNET. The set of metrics applied to the comparison is Recall-Oriented Understudy for Gisting Evaluation (ROUGE). The timer with CPU 1.6 GHZ is included to assess the algorithmic efficiency. Both algorithms are transferred learnings. Transformers are known for solving various Natural Language Processing (NLP) tasks such as text generation and chatbot. What leads to this research is a fundamental question to ask. The Google Research team attempted to develop a different approach to address the inherent self-attention mechanism problem in Transformers models called the block sparsity. The research team used a mathematical assessment to demonstrate the block sparsity that reduced the quadratic dependency to the linear dependency (in relationship of the number of tokens and memory or time) (Zaheer et al., 2020), which is skeptical.</p>

## KEY FINDINGS <a id = 'findings'></a>

<ul>
  <li><p align = 'justify'> As indicated by Randomized Controlled Trial analysis, the Big Bird model performance is significantly higher than XLNET at Bonferroni correction level. </p></li>
  <li><p align = 'justify'>However, XLNET outperforms Big Bird model in memory efficiency based on producing text prediction per article text.</p></li>
  <li><p align = 'justify'>There is evidence that the model produces some redundancies produced by Big Bird text summarization.</p></li>
  <li><p align = 'justify'>Other evidence shows that the Big Bird model reduces quadratic dependency to linear dependency which is against my hypothesis.</p></li>
</ul>

## TRANSFORMERS ARCHITECTURE <a id='transformers'></a>

<p align = 'justify'>To understand what the result of quadratic dependency is, Transformers architecture and its history needs to be addressed first. Before the Transformers architecture is established, the long short-term memory (LSTM) and gated recurrent neural networks are considered as the state-of-the-art approaches in addressing NLP problems. The significant constraint in this model is sequential computation, so the attention mechanisms help to remove the constraint. As a result, the architecture proves to be a milestone in the NLP area throughout the years. </p>

<img src='https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Transformers%20Architecture.png?raw=true'/>

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

<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/bird%20view.png?raw=true'/>

<p align = 'justify'>As seen by above, each color is associated with global, sliding, and random connections while each white slot represents no connection. For example, there is no connection between "work" and "is." This approach reduces time complexity, which comes with the price of no theoretical guarantees as the Google Research acknowledges (Zaheer et al., 2020).</p>

## METHOD <a id ='method'></a> 
<ul>
  <li><p align = 'justify'> As indicated by Randomized Controlled Trial analysis, the Big Bird model performance is significantly higher than XLNET at Bonferroni correction level. </p></li>
  <li><p align = 'justify'>However, XLNET outperforms Big Bird model in memory efficiency based on producing text prediction per article text.</p></li>
  <li><p align = 'justify'>There is evidence that the model produces some redundancies produced by Big Bird text summarization.</p></li>
  <li><p align = 'justify'>On another hand, the evidence shows that the Big Bird model reduces quadratic dependency to linear against my hypothesis.</p></li>
</ul>

## DATASET <a id ='dataset'></a>
""

## ACTIONABLE INSIGHTS <a id ='insights'></a>

<img src = "https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/comparing.gif?raw=true"/>

## FUTURE RESEARCH <a id = 'future'></a>

""

## CONCLUSION <a id='concluding'></a>


## REFERENCES <a id = 'refer'></a>

https://papers.nips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf
https://arxiv.org/pdf/1904.10509.pdf
https://arxiv.org/pdf/1906.08237v2.pdf


