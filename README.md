# Text-Summarization

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jonahwinninghoff/Springboard/graphs/commit-activity)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**[Overview](#overview)** | **[Findings](#findings)** | **[Transformers](#transformers)** | **[Big Bird](#bigbird)** | **[Method](#method)** | **[Dataset](#dataset)** | **[Insights](#insights)** | **[Future](#future)** | **[Conclusion](#concluding)** | **[References](#refer)**

## OVERVIEW <a id='overview'></a>

<p align = 'justify'> For this research, two text summarizations are compared using a specific metrics and a timer. Two text summarizations outlined in this research are the Big Bird and XLNET. The set of metrics applied to the comparison is Recall-Oriented Understudy for Gisting Evaluation (ROUGE). The timer with CPU 1.6 GHZ is included to assess the algorithmic efficiency. Both algorithms are transferred learnings. Transformers are known for solving various Natural Language Processing (NLP) tasks such as text generation and chatbot. What leads to this research is a fundamental question to ask. The Google Research team attempts to develop a different approach to address the inherent self-attention mechanism problem in Transformers models called the block sparsity. This research team uses a mathematical assessment to demonstrate the block sparsity that helps reduce this quadratic dependency to linear (in relationship of the number of tokens and memory or time) (Zaheer et al., 2020), which is skeptical.</p>

## KEY FINDINGS <a id = 'findings'></a>

<ul>
  <li><p align = 'justify'> As indicated by Randomized Controlled Trial analysis, the Big Bird model performance is significantly higher than XLNET at Bonferroni correction level. </p></li>
  <li><p align = 'justify'>However, XLNET outperforms Big Bird model in memory efficiency based on producing text prediction per article text.</p></li>
  <li><p align = 'justify'>There is evidence that the model produces some redundancies produced by Big Bird text summarization.</p></li>
  <li><p align = 'justify'>On another hand, the evidence shows that the Big Bird model reduces quadratic dependency to linear against my hypothesis.</p></li>
</ul>

## TRANSFORMERS ARCHITECTURE <a id='transformers'></a>

<p align = 'justify'>To understand what is the result of quadratic dependency, Transformers architecture and its history must be addressed first. Before the Transformers architecture is established, the long short-term memory (LSTM) and gated recurrent neural networks are considered as the state-of-the-art approaches in addressing NLP problems. The significant constraint in this model is sequential computation, so the attention mechanisms help to remove the constraint. As a result, the architecture proves to be a milestone in the NLP area throughout the years. </p>

<img src='https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Transformers%20Architecture.png?raw=true'/>

<p align = 'justify'>As indicated by above, the architecture consists of two multi-layered parts, which are encoder and decoder. Both processings absorb word embeddings processed by word2vec. The difference between encoder and decoder is relatively straightforward. The representation of encoder is X (x<sub>1</sub>, ..., x<sub>n</sub>) while the representation of decoder is Z (z<sub>1</sub>, ..., z<sub>n</sub>). In this case, the representation of encoder is word embedding of unsummarized text while the representation of decoder is that of the actual summarized text.</p>

<p align = 'justify'>The self-attention is associated with <i>head<sub>i</sub></i>, which is associated with <i>multi-head attention</i> as seen by above and below. The self-attention contains a matrix of a set of queries multiplied by transposed key divided by a root square of key dimensionality in softmax function multiplied by values. The softmax is a generalized version of logistic function. The values can be considered as weight that can be updated in a light of new information. This self-attention is built on matrix multiplicaiton code in order to be more space-efficient and faster.</p>

<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Attention-formula.png?raw=true'/>
<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/head_i.png?raw=true'/>
<img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/multi-head-attention.png?raw=true'/>

<p align = 'justify'> The <i>head<sub>i</sub></i> has a vector of Q, K, and V each multplied by a vector of weights while the multi-head attention is simply a concatenation of <i>head<sub>i</sub></i> (Vaswani et al., 2017). Based on the graph theory, the problem is that the attention mechansim has quadratic dependency due to the fully connected graph (Zaheer et al., 2020). This is known to be <i>sparsification problem</i> (Zaheer et al., 2020).</p>

## BIG BIRD <a id ='bigbird'></a>

""

## METHOD <a id ='method'></a> 
""

## DATASET <a id ='dataset'></a>
""

## ACTIONABLE INSIGHTS <a id ='insights'></a>

""

## FUTURE RESEARCH <a id = 'future'></a>

""

## CONCLUSION <a id='concluding'></a>


## REFERENCES <a id = 'refer'></a>

""

