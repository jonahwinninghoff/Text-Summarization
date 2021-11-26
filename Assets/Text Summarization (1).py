#!/usr/bin/env python
# coding: utf-8

# <p style="color:#e8574d; font-size:30px; font-weight:bold; font-family:Helvetica"> Content </p>
# 
# <ul style="font-size:16px; font-family:Helvetica">
#     <li><a href='#copyright' style="color:#696969; text-decoration: none">Copyright and License</a></li>
#     <li><a href='#modules' style="color:#696969; text-decoration: none">All Necessary Modules and Datasets</a></li>
#     <li><a href='#eda' style="color:#696969; text-decoration: none">Exploratory Data Analysis</a></li>
#     <li><a href='#timer' style="color:#696969; text-decoration: none">Time Complexity: Big Bird and Transformers</a></li>
#     <li><a href='#cited' style="color:#696969; text-decoration: none">Reference</a></li>
# </ul> 
# 
# 

# <p style="color:#e8574d; font-size:20px; font-weight:bold; font-family:Helvetica">Copyright and License<a id='copyright'></a></p>

# <p style="color:#696969; font-size:14px">License and copyright notice</p>

# In[1]:


get_ipython().system('pip show bigbird')


# <p style="color:#696969; font-size:14px; margin:0px; padding:0px">State changes:</p>
# <p style='font-family:monospace; margin-left:20px'>Name:</p>

# <p style="color:#e8574d; font-size:20px; font-weight:bold; font-family:Helvetica">All Necessary Modules and Datasets<a id='module'></a></p>

# In[2]:


# Transformers and BigBird
import tensorflow as tf
from summarizer import Summarizer,TransformerSummarizer

# Reading and manipulate NLP data
import math
import numpy as np
from nltk.corpus import words
import pandas as pd
from random import sample
import re
import tensorflow.compat.v2 as tf
from tqdm import tqdm

# Plot
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

# Evaluate metrics
from rouge_score import rouge_scorer
import time

# Tensorflow dataset
import tensorflow_datasets as tfds
import tensorflow_text as tft


# In[3]:


# Clean text up
def clean_up_text(thedict):
    for i in range(len(thedict['abstract_text'])):
        # Remove all <S> and </S> and double spaces and ", " in abstract text
        thedict['abstract_text'][i] = re.sub(' +', ' ', thedict['abstract_text'][i].replace('<S>','').replace(
        '</S>','').replace('", "',''))
        thedict['article_text'][i] = re.sub(' +', ' ', thedict['article_text'][i].replace(
            '", "','').replace('["',''))
    return thedict

# Create reading arxiv data for this particular txt problem
def read_arxiv_data(directory):
    with open(directory) as f: # read lines
        lines = f.readlines()
    article_id = []            # Instantiate empty lists
    article_text = []
    abstract_text = []
    for i in range(len(lines)): # for loop to find start and end strings
        article_id.append(lines[i][lines[i].find('"article_id": "')+len(
        '"article_id": "'):lines[i].find('",')])
        article_text.append(lines[i][lines[i].find('"article_text": ')+len(
        '"article_text": '):lines[i].find('"], ')])
        abstract_text.append(lines[i][lines[i].find('"abstract_text": ["<S>')+len(
            'abstract_text": ["<S>  '):lines[i].find(' </S>"]')])
    thedict = {'article_id':article_id,'article_text':article_text,'abstract_text':abstract_text} 
    return clean_up_text(thedict) # Use clean up tool 


# In[4]:


# Read validation set
result = read_arxiv_data('arxiv-dataset/val.txt')


# <p style="color:#e8574d; font-size:20px; font-weight:bold; font-family:Helvetica">Exploratory Data Analysis<a id='eda'></a></p>

# In[5]:


np.random.seed(121)
r = int(np.random.choice(range(len(result['article_id'])), size=1))


# In[6]:


print(result['article_id'][r])


# In[7]:


print(result['abstract_text'][r])


# In[8]:


print(result['article_text'][r])


# In[9]:


# Create the Standard Scaler for dictionary
def z_score(thedict):
    if type(thedict) == type({}): # If dict type
        adict = thedict.copy()
        thedict = list(thedict.values())
    else:
        adict = None
    u = np.mean(thedict)
    s = np.std(thedict)
    x = thedict
    Z = (x - u)/s
    if type(adict) == type({}): # If dict type
        # Instantiate i
        i = 0
        for k in adict.keys():
            adict[k] = Z[i]
            i+=1 
        return adict
    else:             # If not dict type
        return Z


# In[10]:


# Check for abstract text sampling convergence
convergence = {}
for i in range(10,3000):
    sampling = sample(result['abstract_text'],i)
    convergence[i] = sum(map(len, sampling))/float(len(sampling))
convergence = z_score(convergence)
# Check for article text sampling convergence
convergence1 = {}
for i in range(10,3000):
    sampling = sample(result['article_text'],i)
    convergence1[i] = sum(map(len, sampling))/float(len(sampling))
convergence1 = z_score(convergence1)


# In[11]:


plt.style.use('fivethirtyeight')
plt.figure(figsize = (15,10))
plt.plot(convergence.keys(),convergence.values(),alpha=0.5,label = 'abstract')
plt.plot(convergence1.keys(),convergence1.values(),alpha=0.5, label = 'article')
plt.xlabel('Sampling size')
plt.ylabel('Mean number of characters with spaces in text (in z-score)')
plt.title('Convergence')
plt.legend()
plt.show()


# In[12]:


# Select 1500 sampling size
sampling = sample(result['abstract_text'],1500)
sampling1 = sample(result['article_text'],1500)


# In[13]:


ecdf = ECDF(list(map(len,sampling)))
ecdf1 = ECDF(list(map(len,sampling1)))

# plot the ecdf
fig, ax = plt.subplots(1,2,figsize = (15,7))
ax[0].plot(ecdf.x, ecdf.y, label = 'abstract text')
ax[1].plot(ecdf1.x, ecdf1.y, color = 'red', label = 'article text')
fig.suptitle('Empirical Cumulative Distribution Function')
fig.text(0.5, 0.00, 'Number of characters with space in text', ha='center')
fig.legend()
plt.show()


# In[14]:


def str_info(thelist):
    print('length of text')
    print('mean:', round(sum(map(len,thelist))/len(thelist),2))
    print('median:', round(np.median(list(map(len,thelist))),2))
    print('stdev:', round(np.std(list(map(len,thelist))),2))
    print('0.975:', round(np.quantile(list(map(len,thelist)),0.975),2))
    print('0.025:', round(np.quantile(list(map(len,thelist)),0.025),2))


# In[15]:


str_info(sampling)


# In[16]:


str_info(sample(result['article_text'],1500))


# In[17]:


def english_match(alist):
    thelist = [] # instantiate list
    for i in range(len(alist)):  # tokenize
        thelist.extend(alist[i].split())
    # Apply numpy for more efficiency and check if tokenizer matches dictionary
    # and set the benchmark for english percentage
    return str(round(100*np.mean(np.isin(np.array(thelist),
           np.array(words.words())).astype(int)),2))+'%'


# In[18]:


print('The percentage of abstract texts that matches NLTK dictionaries:',
      english_match(sample(result['abstract_text'],1500)))
print('The percentage of article texts that matches NLTK dictionaries:',
      english_match(sample(result['article_text'],100)))


# <p style="color:#e8574d; font-size:20px; font-weight:bold; font-family:Helvetica">Review on Transformers and Big Bird under the Hood<a id='hood'></a></p>

# <p style="color:#696969; font-size:14px">Transformers</p>

# <img src='https://github.com/jonahwinninghoff/Text-Summarization/raw/main/Images/Transformers%20Architecture.png'/>

# <p style="color:#696969; font-size:14px; text-align: justify">The left side of transfomers is encoder and another side decoder. Both encoder and decoder absorb information from word embeddings processed by word2vec. The difference is that the representation of encoder is X while the representation of decoder is Z (Vaswani et al., 2017). For this analysis, the representation of encoder is word embedding of <i>article_text</i> that of decoder word embedding of <i>abstract_text</i>.</p>

# <img src='https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Attention-formula.png?raw=true'/>

# <p style="color:#696969; font-size:14px;text-align: justify">The self attention is also known as scaled dot-product attention, which is associated with multi-head attention. The K represents of keys while the V represents values. The Q is a matrix of a set of queries. The query refers to particular word that has has posed (i.e., "You") while the key refer to next word (i.e., "have"). The d<sub>k</sub> is the key dimensionality. The softmax is a generalization of logistic function.</p>

# <img src='https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/head_i.png?raw=true'/>

# <p style="color:#696969; font-size:14px">The head<sub>i</sub> is with a vector of Q, K, and V matrically multiplied by a vector of weights</p>

# <img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/multi-head-attention.png?raw=true'/>

# <p style="color:#696969; font-size:14px">The multi-head attention is a concatenation of head_i multiplied by a weight (Vaswani et al., 2017).</p>

# <p style="color:#696969; font-size:14px">Big Bird</p>

# <p style="color:#696969; font-size:14px;text-align: justify">The Big Bird algorithm is almost identical to transformers but what is changed in this algorithm is that block sparse attention is in use. The objective in this algorithm is to ensure that algorithm is more efficient using three different connections instead of normal attention that relies on full connection. These connections are global, sliding, and random. For example:</p>

# <img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Block%20Sparse.png?raw=true' />

# <img src = 'https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/Compare%20attentions.png?raw=true' />

# <img src='https://github.com/jonahwinninghoff/Text-Summarization/blob/main/Images/bird%20view.png?raw=true'/>

# <p style="color:#696969; font-size:14px">(Gupta, 2021)</p>

# <p style="color:#696969; font-size:14px;text-align: justify">Both figures are pretty self-explained. However, the Big Bird creators claim that this algorithm is linear dependence for sequence. This is why this analysis is to evaluate to determine if this claim is true.</p>

# <p style="color:#e8574d; font-size:20px; font-weight:bold; font-family:Helvetica">Time Complexity: Big Bird and Transformers<a id='timer'></a></p>

# In[19]:


# Create new column for word counts
result['abstract word counts'] = [len(result['abstract_text'][0].split())]
for i in range(1,len(result['article_id'])):
    result['abstract word counts'].append(len(result['abstract_text'][i].split()))


# In[20]:


# Develop complete assessment on Transformer text summarization
def assess_transf(data,size_of_sampling = 100, seed = 0, warning = False):
    thelist = {} # Instantiate the list in dict form
    
    # Instantiate the Transformer Summarizer and rouge scorer
    model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    
    # Set seed and use size_of_sampling for tuning
    np.random.seed(seed)
    r = np.random.choice(range(len(data['article_id'])), size = size_of_sampling, replace = False)
    
    # Instantiate all variables for dict append
    thelist['article_id'] = []
    thelist['article_text'] = []
    thelist['abstract_text'] = []
    thelist['article word counts'] = []
    thelist['abstract word counts'] = []
    thelist['predicted summary'] = []
    thelist['predicted summary word counts'] = []
    thelist['time in seconds'] = []
    thelist['rouge1'] = []
    thelist['rouge2'] = []
    thelist['rougeL'] = []
    
    # for loop based on size_of_sampling
    for i in range(size_of_sampling):
        # Get data
        thelist['article_id'].append(data['article_id'][r[i]])
        thelist['article_text'].append(data['article_text'][r[i]])
        thelist['article word counts'].append(len(data['article_text'][r[i]].split()))
        thelist['abstract_text'].append(data['abstract_text'][r[i]])
        thelist['abstract word counts'].append(data['abstract word counts'][r[i]])
        
        # Predict summaries with start timer
        start = time.time()
        auto = ''.join(model(data['article_text'][r[i]], 
                             max_length=data['abstract word counts'][r[i]]+20))
        thelist['predicted summary'].append(auto)
        thelist['time in seconds'].append(time.time()-start)
        thelist['predicted summary word counts'].append(len(auto.split()))
        
        # Create score for each summary using rouge as a set of metrics
        scores = scorer.score(data['abstract_text'][r[i]],auto)
        thelist['rouge1'].append(scores['rouge1'][2])
        thelist['rouge2'].append(scores['rouge2'][2])
        thelist['rougeL'].append(scores['rougeL'][2])
    
    return pd.DataFrame(thelist)


# In[21]:


# Develop complete assessment on Transformer text summarization
def assess_bigbir(dict_data,size_of_sampling = 1, seed = 0, warning = False):
    tf.enable_v2_behavior()
    
    # Instantiate the list
    thelist = {}
    
    # Check if dict data type
    if type(dict_data) == type({}):
        # Count words in each article text
        dict_data['article word counts'] = [len(dict_data['article_text'][0].split())]
        for i in range(1,len(result['article_id'])):
            dict_data['article word counts'].append(len(dict_data['article_text'][i].split()))
            
        # Form this data into pandas dataframe
        df = pd.DataFrame(dict_data)
        
        # Set seed and use size_of_sampling for tuning and randomly sample some of data
        np.random.seed(seed)
        r = np.random.choice(range(len(dict_data['article_id'])), size = size_of_sampling, replace = False)
        df = df.iloc[r,:]
        df.reset_index(drop=True, inplace=True) # Reset index
        
        # Random sampling on pandas and form another data called TensorSliceDataset using smaller data
        dataset = tf.data.Dataset.from_tensor_slices(df[['article_id','article_text','abstract_text']])

        # Instantiate the Big Bird Summarizer and rouge scorer
        import warnings
        warnings.filterwarnings('ignore')

        path = 'gs://bigbird-transformer/summarization/pubmed/roberta/saved_model'
        imported_model = tf.saved_model.load(path, tags='serve')
        summarize = imported_model.signatures['serving_default']
        scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        
        # Instantiate all variables for dict append
        thelist['predicted summary'] = []
        thelist['predicted summary word counts'] = []
        thelist['time in seconds'] = []
        thelist['rouge1'] = []
        thelist['rouge2'] = []
        thelist['rougeL'] = []
        
        ## for loop based on size_of_sampling 
        for ex in tqdm(dataset.take(size_of_sampling), position=0):
            start = time.time() # timer
            predicted_summary = summarize(ex[2])['pred_sent'][0] # prediction
            thelist['time in seconds'].append(time.time()-start) # end timer
            score = scorer.score(ex[2].numpy().decode('utf-8'), predicted_summary.numpy().decode('utf-8')) # scores
            thelist['predicted summary'].append(predicted_summary.numpy().decode('utf-8')) # summary
            thelist['predicted summary word counts']= len(predicted_summary.numpy().decode('utf-8').split()) # counts
            thelist['rouge1'].append(score['rouge1'][2]) # rouge1
            thelist['rouge2'].append(score['rouge2'][2]) # rouge2
            thelist['rougeL'].append(score['rougeL'][2]) # rougeL
            
        return pd.concat([df, pd.DataFrame(thelist)],axis=1)
        
    else:
        print('Please use dictionary data type.')


# In[22]:


rouges = assess_bigbir(result,size_of_sampling = 110, seed = 33, warning = False)


# In[23]:


transformer = assess_transf(result, size_of_sampling = 110, seed = 33)


# In[24]:


rouges['block sparsity'] = 1
transformer['block sparsity'] = 0


# In[25]:


pd.concat([rouges, transformer],axis=0).to_csv('bigbirvtrans')


# In[26]:


pd.concat([rouges, transformer],axis=0)


# <p style="color:#e8574d; font-size:20px; font-weight:bold; font-family:Helvetica">Useful Reference<a id='cited'></a></p>

# - https://huggingface.co/blog/big-bird
# - https://www.tensorflow.org/datasets/catalog/scientific_papers
# - http://github.com/google-research/bigbird
# - https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
