# Assignment 2: Transformer Summarizer

Welcome to the second assignment of course 4. In this assignment you will explore summarization using the transformer model. Yes, you will implement the transformer decoder from scratch, but we will slowly walk you through it. There are many hints in this notebook so feel free to use them as needed. 

<img src = "transformerNews.png">



## Outline

- [Introduction](#0)
- [Part 1: Importing the dataset](#1)
    - [1.1 Encode & Decode helper functions](#1.1)
    - [1.2 Defining parameters](#1.2)
    - [1.3 Exploring the data](#1.3)
- [Part 2: Summarization with transformer](#2)
    - [2.1 Dot product attention](#2.1)
        - [Exercise 01](#ex01)
    - [2.2 Causal Attention](#2.2)
        - [Exercise 02](#ex02)
    - [2.3 Transformer decoder block](#2.3)
        - [Exercise 03](#ex03)
    - [2.4 Transformer Language model](#2.4)
        - [Exercise 04](#ex04)
- [Part 3: Training](#3)
    - [3.1 Training the model](#3.1)
        - [Exercise 05](#ex05)
- [Part 4: Evaluation](#4)
    - [4.1 Loading in a trained model](#4.1)
- [Part 5: Testing with your own input](#5) 
    - [Exercise 6](#ex06)
    - [5.1 Greedy decoding](#5.1)
        - [Exercise 07](#ex07)

<a name='0'></a>
### Introduction

Summarization is an important task in natural language processing and could be useful for a consumer enterprise. For example, bots can be used to scrape articles, summarize them, and then you can use sentiment analysis to identify the sentiment about certain stocks. Anyways who wants to read an article or a long email today, when you can build a transformer to summarize text for you. Let's get started, by completing this assignment you will learn to:  

- Use built-in functions to preprocess your data
- Implement DotProductAttention
- Implement Causal Attention
- Understand how attention works
- Build the transformer model
- Evaluate your model
- Summarize an article

As you can tell, this model is slightly different than the ones you have already implemented. This is heavily based on attention and does not rely on sequences, which allows for parallel computing. 


```python
import sys
import os

import numpy as np

import textwrap
wrapper = textwrap.TextWrapper(width=70)

import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp

# to print the entire np array
np.set_printoptions(threshold=sys.maxsize)
```

    INFO:tensorflow:tokens_length=568 inputs_length=512 targets_length=114 noise_density=0.15 mean_noise_span_length=3.0 
    

<a name='1'></a>
## Part 1: Importing the dataset

Trax makes it easy to work with Tensorflow's datasets:


```python
# This will download the dataset if no data_dir is specified.
# Downloading and processing can take bit of time,
# so we have the data already in 'data/' for you

# Importing CNN/DailyMail articles dataset
train_stream_fn = trax.data.TFDS('cnn_dailymail',
                                 data_dir='data/',
                                 keys=('article', 'highlights'),
                                 train=True)

# This should be much faster as the data is downloaded already.
eval_stream_fn = trax.data.TFDS('cnn_dailymail',
                                data_dir='data/',
                                keys=('article', 'highlights'),
                                train=False)
```

<a name='1.1'></a>
## 1.1 Tokenize & Detokenize helper functions

Just like in the previous assignment, the cell above loads in the encoder for you. Given any data set, you have to be able to map words to their indices, and indices to their words. The inputs and outputs to your [Trax](https://github.com/google/trax) models are usually tensors of numbers where each number corresponds to a word. If you were to process your data manually, you would have to make use of the following: 

- <span style='color:blue'> word2Ind: </span> a dictionary mapping the word to its index.
- <span style='color:blue'> ind2Word:</span> a dictionary mapping the index to its word.
- <span style='color:blue'> word2Count:</span> a dictionary mapping the word to the number of times it appears. 
- <span style='color:blue'> num_words:</span> total number of words that have appeared. 

Since you have already implemented these in previous assignments of the specialization, we will provide you with helper functions that will do this for you. Run the cell below to get the following functions:

- <span style='color:blue'> tokenize: </span> converts a text sentence to its corresponding token list (i.e. list of indices). Also converts words to subwords.
- <span style='color:blue'> detokenize: </span> converts a token list to its corresponding sentence (i.e. string).


```python
def tokenize(input_str, EOS=1):
    """Input str to features dict, ready for inference"""
  
    # Use the trax.data.tokenize method. It takes streams and returns streams,
    # we get around it by making a 1-element stream with `iter`.
    inputs =  next(trax.data.tokenize(iter([input_str]),
                                      vocab_dir='vocab_dir/',
                                      vocab_file='summarize32k.subword.subwords'))
    
    # Mark the end of the sentence with EOS
    return list(inputs) + [EOS]

def detokenize(integers):
    """List of ints to str"""
  
    s = trax.data.detokenize(integers,
                             vocab_dir='vocab_dir/',
                             vocab_file='summarize32k.subword.subwords')
    
    return wrapper.fill(s)
```

<a name='1.2'></a>

## 1.2 Preprocessing for Language Models: Concatenate It!

This week you will use a language model -- Transformer Decoder -- to solve
an input-output problem. As you know, language models only predict the next
word, they have no notion of inputs. To create a single input suitable for
a language model, we concatenate inputs with targets putting a separator
in between. We also need to create a mask -- with 0s at inputs and 1s at targets -- so that the model is not penalized for mis-predicting the article and only focuses on the summary. See the preprocess function below for how this is done.


```python
# Special tokens
SEP = 0 # Padding or separator token
EOS = 1 # End of sentence token

# Concatenate tokenized inputs and targets using 0 as separator.
def preprocess(stream):
    for (article, summary) in stream:
        joint = np.array(list(article) + [EOS, SEP] + list(summary) + [EOS])
        mask = [0] * (len(list(article)) + 2) + [1] * (len(list(summary)) + 1) # Accounting for EOS and SEP
        yield joint, joint, np.array(mask)

# You can combine a few data preprocessing steps into a pipeline like this.
input_pipeline = trax.data.Serial(
    # Tokenizes
    trax.data.Tokenize(vocab_dir='vocab_dir/',
                       vocab_file='summarize32k.subword.subwords'),
    # Uses function defined above
    preprocess,
    # Filters out examples longer than 2048
    trax.data.FilterByLength(2048)
)

# Apply preprocessing to data streams.
train_stream = input_pipeline(train_stream_fn())
eval_stream = input_pipeline(eval_stream_fn())

train_input, train_target, train_mask = next(train_stream)

assert sum((train_input - train_target)**2) == 0  # They are the same in Language Model (LM).
```


```python
# prints mask, 0s on article, 1s on summary
print(f'Single example mask:\n\n {train_mask}')
```

    Single example mask:
    
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    


```python
# prints: [Example][<EOS>][<pad>][Example Summary][<EOS>]
print(f'Single example:\n\n {detokenize(train_input)}')
```

    Single example:
    
     By . Associated Press . PUBLISHED: . 14:11 EST, 25 October 2013 . | .
    UPDATED: . 15:36 EST, 25 October 2013 . The bishop of the Fargo
    Catholic Diocese in North Dakota has exposed potentially hundreds of
    church members in Fargo, Grand Forks and Jamestown to the hepatitis A
    virus in late September and early October. The state Health Department
    has issued an advisory of exposure for anyone who attended five
    churches and took communion. Bishop John Folda (pictured) of the Fargo
    Catholic Diocese in North Dakota has exposed potentially hundreds of
    church members in Fargo, Grand Forks and Jamestown to the hepatitis A
    . State Immunization Program Manager Molly Howell says the risk is
    low, but officials feel it's important to alert people to the possible
    exposure. The diocese announced on Monday that Bishop John Folda is
    taking time off after being diagnosed with hepatitis A. The diocese
    says he contracted the infection through contaminated food while
    attending a conference for newly ordained bishops in Italy last month.
    Symptoms of hepatitis A include fever, tiredness, loss of appetite,
    nausea and abdominal discomfort. Fargo Catholic Diocese in North
    Dakota (pictured) is where the bishop is located .<EOS><pad>BishopJohn
    Folda, of North Dakota, is taking time off after being diagnosed . He
    contracted the infection through contaminated food in Italy . Church
    members in Fargo, Grand Forks and Jamestown could have been exposed
    .<EOS>
    

<a name='1.3'></a>

## 1.3 Batching with bucketing

As in the previous week, we use bucketing to create batches of data.


```python
# Bucketing to create batched generators.

# Buckets are defined in terms of boundaries and batch sizes.
# Batch_sizes[i] determines the batch size for items with length < boundaries[i]
# So below, we'll take a batch of 16 sentences of length < 128 , 8 of length < 256,
# 4 of length < 512. And so on. 
boundaries =  [128, 256,  512, 1024]
batch_sizes = [16,    8,    4,    2, 1]

# Create the streams.
train_batch_stream = trax.data.BucketByLength(
    boundaries, batch_sizes)(train_stream)

eval_batch_stream = trax.data.BucketByLength(
    boundaries, batch_sizes)(eval_stream)
```


```python
# Every execution will result in generation of a different article
# Try running this cell multiple times to see how the length of the examples affects the batch size
input_batch, _, mask_batch = next(train_batch_stream)

# Shape of the input_batch
input_batch.shape
```




    (1, 1201)




```python
# print corresponding integer values
print(input_batch[0])
```

    [   27 23176  4694  1779  1343    28   506  1091   132    28   570     6
        78  7124   192 14454    15  3570  2067    23    46 26133    17  1019
       635    91     3  5349 23421   494     6 10487     2   728     2  1353
      3156   278  1838    28   736   809    28 13481  7511    22   625    28
      1311  2396     3   187    22  1353  1510   181 16146  1049   320   103
         2    22 26563   651   467   213   826   192  3156  1262    28 13131
         4   186 16949    17    71 12319  6604   828 29725     4     5  1081
      1083   213    54   138     3  5349 23421   494     6 10487     2   728
         8   346    12  1353   354    15  3570  2067  7511    22 24497   570
         6    78    71   213  1081   144  3360   691 12319  6604   828     2
       705     8   231    24   305   710   272  1838    68  6341   379     9
       570     6    78  7124   436   219   132   560   429     3   368 23421
       494     6 10487     7     5  1081  1353 10874 20919   217     8 12370
        21    12  2713   127 23421   494     6 10487    40 23176   809   518
       150   181   290  3892   275   527  8947   171  1269   936   213  9025
         3    69  1353   233  8272   527  6056   583   691  4398  3156   809
     14507  5429   812  7356     3  3622  6604   828     2    28   705     6
       104     6   292 15004   181 29725     4     5 21961  1838 10687    45
         2 11985   527 11907  5364     2    40    43  1383   213  2801  1248
      1078   809    28 13481    35    40    19 23176   116  4016     2   864
       127     3   305  1353  3156 17775 12979  3095   186    77  1353   669
     27439  6050 13459  1628  1290   131   143    18   757   320  2501   213
     25725 29725     2    41   969     3 16978  1822  9855  1962     2 17347
        16     2   127  4601 27439  6050 13459  1628  5349 23421   494     6
     10487 29725     4     5  3156  2868   132   213 15191   583   527    28
       506  1091     2 12319  6604   828     2    28   583   285   143    18
        46 13488 23707  6050 13459  1628   368 23421   494     6 10487   436
       213   884   320  3429    61    15  3570  2067  6715  3156   186     2
       673  1510   181 16146  1049   320   824  1311  2396     2  1353    90
     15438    17   285    22  2214   320 17950    28   346     6   650 13131
         4     2  7228   213  1052   763   314    71   213  2358   527  3622
      6604   828 29725     4     5 18352  2398  1081     3  3622  6604   828
      1353  7214   213 19839   277   527    68 27439  9275  1628 12320  5403
      9242  5590  2385    35   710   272  1838    68  6341   132  2642 11969
     27439  6050 13459  1628  3622  6604   828   669 27884     4    40 27872
       391    28  5302   531  2504   527    68     3   305  1353    43  4925
       278   523  1383   163 20812  2801  1248  1078   186  1353  3156 17775
     12979  3095 23707  6050 13459  1628   305    40  5945   320  1242    68
      1078  7511   131   540   278   320  8916   285   131    40  2362 15627
         3  1561  1078  8075   114   369  1613  1838    68   102    41  7584
        17   458 23707  6050 13459  1628  3622  6604   828 29725     4     5
       583   132    97  2861  6107 17946     5   213  6349   527   354    28
       650     6   475  3570  2067  6715  3156  4172 29725   391  2713    25
      3630   320   245 17388   181  1884  4140  1838 23421   494     6 10487
      1820     2    35   132  4140   329   926   102   213  5556    22  1353
        86 25070   918   155   213  6700     6  2057  3602     3     9  4038
      2256  1248   864   285    22    62    18    46    95   213  3602   809
       213    55    15   651  6866  4604   279  1205  3622  6604   828 29725
         4     5  2498 12320  5403  9242  5590  2385    78    28   826   542
     15902  3569     2 11985   527 11907  5364     2    78   560   253     2
       429     3   405  2067   992  1606    22  1353    43 17997   595   239
       213    55   527   213  7124     3  6753  1565  8120   479     2  1838
     12887 26509 21380   328 29725     4     5  1839 25725  2694  1676     2
       127  3611   871  5784  1435  1248 12319     7     5   228   809   824
        55     3   305    40    46    64  1248  1078   809    28 13481   132
     15010  7301   285  2801     2    35    40    19    40   116  4016  1782
       871  2694  1606   285    77  1353  1290   131   143    18   757   320
      2501   213 25725   186  8075   114   103   919    68    68   177  1782
       368 23421   494     6 10487    40   346   126   132 15902  3569   186
      1326  1248  1078   809    28 13481  4872    22  6005  6929   809   518
       150   320   290  3892   275   527  7468    81     3    69 12402     7
        26   209   346   213 13481   320   955   278  7511   213 25725  1841
       809   239   128    10  3229  2535  1782   129  8198     7    26   217
       320   245 17388   181  1884  4140  1838   134  1820   186   849  1884
       576   329   926   102   213 25725  1606    22  1353 25070   918   155
       213  3602     2    51  2253    22    62    18    46    95   213  3602
       809   213    55   527   213 25725   186   132 13040  2398    61   592
         2   213  4038  2256  1782     9   641   527    15  2067   992  1606
       285    22  1353 17997   595    78    15  2067   239   213    55   527
       213 25725    90   103     7     5  1232   761   824    62    43    18
      3625   320    15  4398  3156   186  1201   527   490  2002 23421   494
         6 10487  1353   233  8272   527  6056   583   691  4398  3156   355
        28  2145   809 14507  5429   812     8 12370    21    12    69   969
      3611   368 23421   494     6 10487    39   169  3263   635    91   936
      5892     2    35 12319     7     5   228    18   913    68  8232  1782
        13  1525   824    39   191   101   362  3060   171  6642   116  4016
       186  1269   936   213  9025     2   181   354    28  2067   640    41
         7   165    78   213   826  1782     9 26024   527  6700  3156   186
      3156  6715   354    28  3570  2067  1435  3787     3  2994  1779   952
       320   124    90   993  3736    28  3537    55   132  2173     3    56
       347  6335   141  7270 15191   213  4472   527 16972   595    97 23891
      6412    49  1151 20327 27439  6050 13459  1628   368 23421   494     6
     10487    39   169  3263   635    91   936  5892     2    35 12319 29725
         4     5   228    18   913    68  1019   545     3    13  1525   824
        39   191   101   362  3060   171  6642   116  4016   186  1269   936
       213  9025     2   181   354    28  2067   640    41 29725     4   165
        78   213   826     3    56   347  6335   141  7270 15191   213  4472
       527 16972   595    97 23891  6412    49  1151  4172 29725   391 23421
       494     6 10487     2   527 14735     2 11985   527 11907  5364     2
      1353    43 24306  5831  4461  1838  3156  1019  1223    91 27439  9275
      1628   102  1480    22    39    18   320   976   163  2008   165     6
      1166    10     1     0  5349 23421   494     6 10487     2   728     2
        40 23176   809   518   150  3892   275   171  3156  1081 16346 27439
      6774  1628  5670   354  2067  7511    22 26563   651   467   826   132
     15902  3569     2 11985   527 11907  5364 16346 27439  6774  1628  3481
      3094   570     6    78    71   705     6   104     6   292 12319  6604
       828     7     5  1081     2  1779   710   132  2642 16346 27439  6774
      1628  2713   476    22    62    18    46    95   904  6700     6  2057
      3602   809    55   527  7124 16346 27439  6774  1628    69  1353   233
      8272   809 14507  5429   812   527  6056   583   691  4398  3156  2104
         1]
    

Things to notice:
 - First we see the corresponding values of the words.
 - The first 1, which represents the `<EOS>` tag of the article.
 - Followed by a 0, which represents a `<pad>` tag.
 - After the first 0 (`<pad>` tag) the corresponding values are of the words that are used for the summary of the article.
 - The second 1 represents the `<EOS>` tag for the summary.
 - All the trailing 0s represent `<pad>` tags which are appended to maintain consistent length (If you don't see them then it would mean it is already of max length)
 


```python
# print the article and its summary
print('Article:\n\n', detokenize(input_batch[0]))
```

    Article:
    
     A drunk driver who killed a young woman in a head-on crash while
    checking his mobile phone has been jailed for six years. Craig
    Eccleston-Todd, 27, was driving home from a night at a pub when he
    received a text message. As he was reading or replying to it, he
    veered across the road while driving round a bend and smashed into
    Rachel Titley’s car coming the other way. Craig Eccleston-Todd, 27
    (left) was using his mobile phone when he crashed head-on into the car
    being driven by Rachel Titley, 28 (right). She died later from her
    injuries . The head-on crash took place in October 2013. Mr Eccleston-
    Todd's car was barely recognisable (pictured) Police said Eccleston-
    Todd had drunk at least three or four pints of beer before getting
    behind the wheel. He was found guilty of causing death by dangerous
    driving at Portsmouth Crown Court yesterday. Miss Titley, a 28-year-
    old solicitor’s clerk from Cowes, Isle of Wight, had also spent the
    evening with friends at a pub but had not drunk any alcohol, police
    said. She was driving responsibly and there was ‘nothing she could
    have done to avoid the collision’, they added. Lindsay Pennell,
    prosecuting, said: ‘Craig Eccleston-Todd’s driving resulted in the
    tragic death of a young woman, Rachel Titley, a death that could have
    been avoided. ‘Mr Eccleston-Todd took the decision to pick up his
    mobile phone whilst driving and, either reading or replying to this
    text message, was so distracted that he failed to negotiate a left-
    hand bend, crossing the central white line into the path of Miss
    Titley’s oncoming car. Miss Titley was pulled the wreckage of
    her Daihatsu Cuore but died later from her injuries in hospital .
    ‘Miss Titley [had] a bright future ahead of her. She was also
    returning home having spent an enjoyable evening with friends and was
    driving responsibly. ‘She had arranged to contact her friends when she
    got home to confirm that she had arrived safely. Her friends sadly
    never heard from her after they parted company. ‘Miss Titley’s death
    in these circumstances reiterates the danger of using a hand-held
    mobile phone whilst driving.’ Police were unable to take breath or
    blood tests from Eccleston-Todd immediately, but in tests several
    hours after the accident he was only marginally under the drink-drive
    limit. The judge agreed with police that he would have been over the
    limit at the time his red Citroen hit Miss Titley’s blue Daihatsu
    Cuore on a road near Yarmouth, Isle of Wight, on October 11, 2013. His
    phone records showed he was also texting around the time of the crash.
    PC Mark Furse, from Hampshire constabulary’s serious collision
    investigation unit, said: 'Our thoughts are with Rachel's family at
    this time. She had been out with friends at a pub in Shalfleet that
    evening, but had not had any alcohol. 'Our investigation showed that
    there was nothing she could have done to avoid the collision and sadly
    it cost her her life. 'Mr Eccleston-Todd had left work in Yarmouth and
    met with friends at a pub where he drank at least three to four pints
    of lager. He hadn't long left the pub to return home when the
    collision occurred at around 9.30pm. 'We weren't able to take breath
    or blood tests from him immediately and although blood taken several
    hours after the collision showed he was marginally under the limit, we
    maintain he would have been over the limit at the time of the
    collision and in summing up today, the judge agreed. 'The analysis of
    his phone records showed that he was texting on his phone around the
    time of the collision so it's highly likely this would also have
    contributed to his dangerous driving and loss of control.' Eccleston-
    Todd was found guilty of causing death by dangerous driving following
    a trial at Portsmouth Crown Court (pictured) He added: 'Mr Eccleston-
    Todd will now spend six years behind bars, but Rachel's family have
    lost her forever. 'I hope this will make people think twice before
    drinking any alcohol and getting behind the wheel, or using a phone
    once they're on the road. 'The dangers of drink driving and driving
    whilst using a mobile phone are obvious. Those who continue to do so
    risk spending a substantial time in prison. This case highlights just
    how tragic the consequences of committing these offences can be.' ‘Mr
    Eccleston-Todd will now spend six years behind bars, but Rachel’s
    family have lost her for ever. I hope this will make people think
    twice before drinking any alcohol and getting behind the wheel, or
    using a phone once they’re on the road. This case highlights just how
    tragic the consequences of committing these offences can be.’
    Eccleston-Todd, of Newport, Isle of Wight, was also disqualified from
    driving for eight years after which he will have to complete an
    extended re-test.<EOS><pad>CraigEccleston-Todd, 27, had drunk at least
    three pints before driving car . Was using phone when he veered across
    road in Yarmouth, Isle of Wight . Crashed head-on into 28-year-old
    Rachel Titley's car, who died in hospital . Police say he would have
    been over legal drink-drive limit at time of crash . He was found
    guilty at Portsmouth Crown Court of causing death by dangerous driving
    .<EOS>
    

You can see that the data has the following structure:
- <span style='color:blue'> [Article] </span> -> `<EOS>` -> `<pad>` -> <span style='color:blue'> [Article Summary] </span> -> `<EOS>` -> (possibly) multiple `<pad>`

The loss is taken only on the summary using cross_entropy as loss function. 

<a name='2'></a>
# Part 2: Summarization with transformer

Now that we have given you the data generator and have handled the preprocessing for you, it is time for you to build your own model. We saved you some time because we know you have already preprocessed data before in this specialization, so we would rather you spend your time doing the next steps. 

You will be implementing the attention from scratch and then using it in your transformer model. Concretely, you will understand how attention works, how you use it to connect the encoder and the decoder.

<img src="transformer_decoder_zoomin.png">

<a name='2.1'></a>
## 2.1 Dot product attention 

Now you will implement dot product attention which takes in a query, key, value, and a mask. It returns the output. 

<img src ="dotproduct.png">


Here are some helper functions that will help you create tensors and display useful information:
   - `create_tensor`  creates a `jax numpy array` from a list of lists.
   - `display_tensor` prints out the shape and the actual tensor.


```python
def create_tensor(t):
    """Create tensor from list of lists"""
    return jnp.array(t)


def display_tensor(t, name):
    """Display shape and tensor"""
    print(f'{name} shape: {t.shape}\n')
    print(f'{t}\n')
```

Before implementing it yourself, you can play around with a toy example of `dot product attention` without the softmax  operation. Technically it would not be `dot product attention` without the softmax but this is done to avoid giving away too much of the answer and the idea is to display these tensors to give you a sense of how they look like.

The formula for attention is this one:

$$
\text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}+{M}\right) V\tag{1}\
$$

$d_{k}$ stands for the dimension of queries and keys.

The `query`, `key`, `value` and `mask` vectors are provided for this example.

Notice that the masking is done using very negative values that will yield a similar effect to using $-\infty $. 


```python
q = create_tensor([[1, 0, 0], [0, 1, 0]])
display_tensor(q, 'query')
k = create_tensor([[1, 2, 3], [4, 5, 6]])
display_tensor(k, 'key')
v = create_tensor([[0, 1, 0], [1, 0, 1]])
display_tensor(v, 'value')
m = create_tensor([[0, 0], [-1e9, 0]])
display_tensor(m, 'mask')
```

    query shape: (2, 3)
    
    [[1 0 0]
     [0 1 0]]
    
    key shape: (2, 3)
    
    [[1 2 3]
     [4 5 6]]
    
    value shape: (2, 3)
    
    [[0 1 0]
     [1 0 1]]
    
    mask shape: (2, 2)
    
    [[ 0.e+00  0.e+00]
     [-1.e+09  0.e+00]]
    
    

**Expected Output:**
```CPP
query shape: (2, 3)

[[1 0 0]
 [0 1 0]]

key shape: (2, 3)

[[1 2 3]
 [4 5 6]]

value shape: (2, 3)

[[0 1 0]
 [1 0 1]]

mask shape: (2, 2)

[[ 0.e+00  0.e+00]
 [-1.e+09  0.e+00]]

```


```python
q_dot_k = q @ k.T / jnp.sqrt(3)
display_tensor(q_dot_k, 'query dot key')
```

    query dot key shape: (2, 2)
    
    [[0.57735026 2.309401  ]
     [1.1547005  2.8867514 ]]
    
    

**Expected Output:**
```CPP
query dot key shape: (2, 2)

[[0.57735026 2.309401  ]
 [1.1547005  2.8867514 ]]
```


```python
masked = q_dot_k + m
display_tensor(masked, 'masked query dot key')
```

    masked query dot key shape: (2, 2)
    
    [[ 5.7735026e-01  2.3094010e+00]
     [-1.0000000e+09  2.8867514e+00]]
    
    

**Expected Output:**
```CPP
masked query dot key shape: (2, 2)

[[ 5.7735026e-01  2.3094010e+00]
 [-1.0000000e+09  2.8867514e+00]]
```


```python
display_tensor(masked @ v, 'masked query dot key dot value')
```

    masked query dot key dot value shape: (2, 3)
    
    [[ 2.3094010e+00  5.7735026e-01  2.3094010e+00]
     [ 2.8867514e+00 -1.0000000e+09  2.8867514e+00]]
    
    

**Expected Output:**
```CPP
masked query dot key dot value shape: (2, 3)

[[ 2.3094010e+00  5.7735026e-01  2.3094010e+00]
 [ 2.8867514e+00 -1.0000000e+09  2.8867514e+00]]
```

In order to use the previous dummy tensors to test some of the graded functions, a batch dimension should be added to them so they mimic the shape of real-life examples. The mask is also replaced by a version of it that resembles the one that is used by trax:


```python
q_with_batch = q[None,:]
display_tensor(q_with_batch, 'query with batch dim')
k_with_batch = k[None,:]
display_tensor(k_with_batch, 'key with batch dim')
v_with_batch = v[None,:]
display_tensor(v_with_batch, 'value with batch dim')
m_bool = create_tensor([[True, True], [False, True]])
display_tensor(m_bool, 'boolean mask')
```

    query with batch dim shape: (1, 2, 3)
    
    [[[1 0 0]
      [0 1 0]]]
    
    key with batch dim shape: (1, 2, 3)
    
    [[[1 2 3]
      [4 5 6]]]
    
    value with batch dim shape: (1, 2, 3)
    
    [[[0 1 0]
      [1 0 1]]]
    
    boolean mask shape: (2, 2)
    
    [[ True  True]
     [False  True]]
    
    

**Expected Output:**
```CPP
query with batch dim shape: (1, 2, 3)

[[[1 0 0]
  [0 1 0]]]

key with batch dim shape: (1, 2, 3)

[[[1 2 3]
  [4 5 6]]]

value with batch dim shape: (1, 2, 3)

[[[0 1 0]
  [1 0 1]]]

boolean mask shape: (2, 2)

[[ True  True]
 [False  True]]
```

<a name='ex01'></a>
### Exercise 01

**Instructions:** Implement the dot product attention. Concretely, implement the following equation


$$
\text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}+{M}\right) V\tag{1}\
$$

$Q$ - query, 
$K$ - key, 
$V$ - values, 
$M$ - mask, 
${d_k}$ - depth/dimension of the queries and keys (used for scaling down)

You can implement this formula either by `trax` numpy (trax.math.numpy) or regular `numpy` but it is recommended to use `jnp`.

Something to take into consideration is that within trax, the masks are tensors of `True/False` values not 0's and $-\infty$ as in the previous example. Within the graded function don't think of applying the mask by summing up matrices, instead use `jnp.where()` and treat the **mask as a tensor of boolean values with `False` for values that need to be masked and True for the ones that don't.**

Also take into account that the real tensors are far more complex than the toy ones you just played with. Because of this avoid using shortened operations such as `@` for dot product or `.T` for transposing. Use `jnp.matmul()` and `jnp.swapaxes()` instead.

This is the self-attention block for the transformer decoder. Good luck!  


```python
# UNQ_C1
# GRADED FUNCTION: DotProductAttention
def DotProductAttention(query, key, value, mask):
    """Dot product self-attention.
    Args:
        query (jax.interpreters.xla.DeviceArray): array of query representations with shape (L_q by d)
        key (jax.interpreters.xla.DeviceArray): array of key representations with shape (L_k by d)
        value (jax.interpreters.xla.DeviceArray): array of value representations with shape (L_k by d) where L_v = L_k
        mask (jax.interpreters.xla.DeviceArray): attention-mask, gates attention with shape (L_q by L_k)

    Returns:
        jax.interpreters.xla.DeviceArray: Self-attention array for q, k, v arrays. (L_q by L_k)
    """

    assert query.shape[-1] == key.shape[-1] == value.shape[-1], "Embedding dimensions of q, k, v aren't all the same"

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # Save depth/dimension of the query embedding for scaling down the dot product
    depth = query.shape[-1] 

    # Calculate scaled query key dot product according to formula above
    dots = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / jnp.sqrt(depth)
    
    # Apply the mask
    if mask is not None: # The 'None' in this line does not need to be replaced
        dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))
    
    # Softmax formula implementation
    # Use trax.fastmath.logsumexp of dots to avoid underflow by division by large numbers
    # Hint: Last axis should be used and keepdims should be True
    # Note: softmax = e^(dots - logsumexp(dots)) = E^dots / sumexp(dots)
    logsumexp = trax.fastmath.logsumexp(dots, axis=-1, keepdims=True)

    # Take exponential of dots minus logsumexp to get softmax
    # Use jnp.exp()
    dots = jnp.exp(dots - logsumexp)

    # Multiply dots by value to get self-attention
    # Use jnp.matmul()
    attention = jnp.matmul(dots, value)

    ## END CODE HERE ###
    
    return attention
```


```python
DotProductAttention(q_with_batch, k_with_batch, v_with_batch, m_bool)
```




    DeviceArray([[[0.8496746 , 0.15032545, 0.8496746 ],
                  [1.        , 0.        , 1.        ]]], dtype=float32)



**Expected Output:**
```CPP
DeviceArray([[[0.8496746 , 0.15032545, 0.8496746 ],
              [1.        , 0.        , 1.        ]]], dtype=float32)
```    

<a name='2.2'></a>

## 2.2 Causal Attention

Now you are going to implement causal attention: multi-headed attention with a mask to attend only to words that occurred before. 

<img src = "causal.png">

In the image above, a word can see everything that is before it, but not what is after it. To implement causal attention, you will have to transform vectors and do many reshapes. You will need to implement the functions below.


<a name='ex02'></a>
### Exercise 02

Implement the following functions that will be needed for Causal Attention:

- <span style='color:blue'> compute_attention_heads </span>: Gets an input $x$ of dimension (batch_size, seqlen, n_heads $\times$ d_head) and splits the last (depth) dimension and stacks it to the zeroth dimension to allow matrix multiplication (batch_size $\times$ n_heads, seqlen, d_head).
- <span style='color:blue'> dot_product_self_attention </span>: Creates a mask matrix with `False` values above the diagonal and `True` values below and calls DotProductAttention which implements dot product self attention.
- <span style='color:blue'> compute_attention_output </span>: Undoes compute_attention_heads by splitting first (vertical) dimension and stacking in the last (depth) dimension (batch_size, seqlen, n_heads $\times$ d_head). These operations concatenate (stack/merge) the heads. 

Next there are some toy tensors which may serve to give you an idea of the data shapes and opperations involved in Causal Attention. They are also useful to test out your functions! 


```python
tensor2d = create_tensor(q)
display_tensor(tensor2d, 'query matrix (2D tensor)')

tensor4d2b = create_tensor([[q, q], [q, q]])
display_tensor(tensor4d2b, 'batch of two (multi-head) collections of query matrices (4D tensor)')

tensor3dc = create_tensor([jnp.concatenate([q, q], axis = -1)])
display_tensor(tensor3dc, 'one batch of concatenated heads of query matrices (3d tensor)')

tensor3dc3b = create_tensor([jnp.concatenate([q, q], axis = -1), jnp.concatenate([q, q], axis = -1), jnp.concatenate([q, q], axis = -1)])
display_tensor(tensor3dc3b, 'three batches of concatenated heads of query matrices (3d tensor)')
```

    query matrix (2D tensor) shape: (2, 3)
    
    [[1 0 0]
     [0 1 0]]
    
    batch of two (multi-head) collections of query matrices (4D tensor) shape: (2, 2, 2, 3)
    
    [[[[1 0 0]
       [0 1 0]]
    
      [[1 0 0]
       [0 1 0]]]
    
    
     [[[1 0 0]
       [0 1 0]]
    
      [[1 0 0]
       [0 1 0]]]]
    
    one batch of concatenated heads of query matrices (3d tensor) shape: (1, 2, 6)
    
    [[[1 0 0 1 0 0]
      [0 1 0 0 1 0]]]
    
    three batches of concatenated heads of query matrices (3d tensor) shape: (3, 2, 6)
    
    [[[1 0 0 1 0 0]
      [0 1 0 0 1 0]]
    
     [[1 0 0 1 0 0]
      [0 1 0 0 1 0]]
    
     [[1 0 0 1 0 0]
      [0 1 0 0 1 0]]]
    
    

It is important to know that the following 3 functions would normally be defined within the `CausalAttention` function further below. 

However this makes these functions harder to test. Because of this, these functions are shown individually using a `closure` (when necessary) that simulates them being inside of the `CausalAttention` function. This is done because they rely on some variables that can be accessed from within `CausalAttention`.

### Support Functions

<span style='color:blue'> compute_attention_heads </span>: Gets an input $x$ of dimension (batch_size, seqlen, n_heads $\times$ d_head) and splits the last (depth) dimension and stacks it to the zeroth dimension to allow matrix multiplication (batch_size $\times$ n_heads, seqlen, d_head).

**For the closures you only have to fill the inner function.**


```python
# UNQ_C2
# GRADED FUNCTION: compute_attention_heads_closure
def compute_attention_heads_closure(n_heads, d_head):
    """ Function that simulates environment inside CausalAttention function.
    Args:
        d_head (int):  dimensionality of heads.
        n_heads (int): number of attention heads.
    Returns:
        function: compute_attention_heads function
    """

    def compute_attention_heads(x):
        """ Compute the attention heads.
        Args:
            x (jax.interpreters.xla.DeviceArray): tensor with shape (batch_size, seqlen, n_heads X d_head).
        Returns:
            jax.interpreters.xla.DeviceArray: reshaped tensor with shape (batch_size X n_heads, seqlen, d_head).
        """
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        
        # Size of the x's batch dimension
        batch_size = x.shape[0]
        # Length of the sequence
        # Should be size of x's first dimension without counting the batch dim
        seqlen = x.shape[1]
        # Reshape x using jnp.reshape()
        # batch_size, seqlen, n_heads*d_head -> batch_size, seqlen, n_heads, d_head
        x = jnp.reshape(x, (batch_size, seqlen, n_heads, d_head))
        # Transpose x using jnp.transpose()
        # batch_size, seqlen, n_heads, d_head -> batch_size, n_heads, seqlen, d_head
        # Note that the values within the tuple are the indexes of the dimensions of x and you must rearrange them
        x = jnp.transpose(x, (0, 2, 1, 3))
        # Reshape x using jnp.reshape()
        # batch_size, n_heads, seqlen, d_head -> batch_size*n_heads, seqlen, d_head
        x = jnp.reshape(x, (-1, seqlen, d_head))
        
        ### END CODE HERE ###
        
        return x
    
    return compute_attention_heads
```


```python
display_tensor(tensor3dc3b, "input tensor")
result_cah = compute_attention_heads_closure(2,3)(tensor3dc3b)
display_tensor(result_cah, "output tensor")
```

    input tensor shape: (3, 2, 6)
    
    [[[1 0 0 1 0 0]
      [0 1 0 0 1 0]]
    
     [[1 0 0 1 0 0]
      [0 1 0 0 1 0]]
    
     [[1 0 0 1 0 0]
      [0 1 0 0 1 0]]]
    
    output tensor shape: (6, 2, 3)
    
    [[[1 0 0]
      [0 1 0]]
    
     [[1 0 0]
      [0 1 0]]
    
     [[1 0 0]
      [0 1 0]]
    
     [[1 0 0]
      [0 1 0]]
    
     [[1 0 0]
      [0 1 0]]
    
     [[1 0 0]
      [0 1 0]]]
    
    

**Expected Output:**
```CPP
input tensor shape: (3, 2, 6)

[[[1 0 0 1 0 0]
  [0 1 0 0 1 0]]

 [[1 0 0 1 0 0]
  [0 1 0 0 1 0]]

 [[1 0 0 1 0 0]
  [0 1 0 0 1 0]]]

output tensor shape: (6, 2, 3)

[[[1 0 0]
  [0 1 0]]

 [[1 0 0]
  [0 1 0]]

 [[1 0 0]
  [0 1 0]]

 [[1 0 0]
  [0 1 0]]

 [[1 0 0]
  [0 1 0]]

 [[1 0 0]
  [0 1 0]]]
```

<span style='color:blue'> dot_product_self_attention </span>: Creates a mask matrix with `False` values above the diagonal and `True` values below and calls DotProductAttention which implements dot product self attention.


```python
# UNQ_C3
# GRADED FUNCTION: dot_product_self_attention
def dot_product_self_attention(q, k, v):
    """ Masked dot product self attention.
    Args:
        q (jax.interpreters.xla.DeviceArray): queries.
        k (jax.interpreters.xla.DeviceArray): keys.
        v (jax.interpreters.xla.DeviceArray): values.
    Returns:
        jax.interpreters.xla.DeviceArray: masked dot product self attention tensor.
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # Hint: mask size should be equal to L_q. Remember that q has shape (batch_size, L_q, d)
    mask_size = q.shape[-2]

    # Creates a matrix with ones below the diagonal and 0s above. It should have shape (1, mask_size, mask_size)
    # Notice that 1's and 0's get casted to True/False by setting dtype to jnp.bool_
    # Use jnp.tril() - Lower triangle of an array and jnp.ones()
    mask = jnp.tril(jnp.ones((1, mask_size, mask_size), dtype=jnp.bool_), k=0)
    
    ### END CODE HERE ###
    
    return DotProductAttention(q, k, v, mask)
```


```python
dot_product_self_attention(q_with_batch, k_with_batch, v_with_batch)
```




    DeviceArray([[[0.        , 1.        , 0.        ],
                  [0.8496746 , 0.15032543, 0.8496746 ]]], dtype=float32)



**Expected Output:**
```CPP
DeviceArray([[[0.        , 1.        , 0.        ],
              [0.8496746 , 0.15032543, 0.8496746 ]]], dtype=float32)
```

<span style='color:blue'> compute_attention_output </span>: Undoes compute_attention_heads by splitting first (vertical) dimension and stacking in the last (depth) dimension (batch_size, seqlen, n_heads $\times$ d_head). These operations concatenate (stack/merge) the heads. 


```python
# UNQ_C4
# GRADED FUNCTION: compute_attention_output_closure
def compute_attention_output_closure(n_heads, d_head):
    """ Function that simulates environment inside CausalAttention function.
    Args:
        d_head (int):  dimensionality of heads.
        n_heads (int): number of attention heads.
    Returns:
        function: compute_attention_output function
    """
    
    def compute_attention_output(x):
        """ Compute the attention output.
        Args:
            x (jax.interpreters.xla.DeviceArray): tensor with shape (batch_size X n_heads, seqlen, d_head).
        Returns:
            jax.interpreters.xla.DeviceArray: reshaped tensor with shape (batch_size, seqlen, n_heads X d_head).
        """
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        
        # Length of the sequence
        # Should be size of x's first dimension without counting the batch dim
        seqlen = x.shape[1]
        # Reshape x using jnp.reshape() to shape (batch_size, n_heads, seqlen, d_head)
        x = jnp.reshape(x, ( -1, n_heads, seqlen, d_head))
        # Transpose x using jnp.transpose() to shape (batch_size, seqlen, n_heads, d_head)
        x = jnp.transpose(x, ( 0, 2, 1 , 3))
        
        ### END CODE HERE ###
        
        # Reshape to allow to concatenate the heads
        return jnp.reshape(x, (-1, seqlen, n_heads * d_head))
    
    return compute_attention_output
```


```python
display_tensor(result_cah, "input tensor")
result_cao = compute_attention_output_closure(2,3)(result_cah)
display_tensor(result_cao, "output tensor")
```

    input tensor shape: (6, 2, 3)
    
    [[[1 0 0]
      [0 1 0]]
    
     [[1 0 0]
      [0 1 0]]
    
     [[1 0 0]
      [0 1 0]]
    
     [[1 0 0]
      [0 1 0]]
    
     [[1 0 0]
      [0 1 0]]
    
     [[1 0 0]
      [0 1 0]]]
    
    output tensor shape: (3, 2, 6)
    
    [[[1 0 0 1 0 0]
      [0 1 0 0 1 0]]
    
     [[1 0 0 1 0 0]
      [0 1 0 0 1 0]]
    
     [[1 0 0 1 0 0]
      [0 1 0 0 1 0]]]
    
    

**Expected Output:**
```CPP
input tensor shape: (6, 2, 3)

[[[1 0 0]
  [0 1 0]]

 [[1 0 0]
  [0 1 0]]

 [[1 0 0]
  [0 1 0]]

 [[1 0 0]
  [0 1 0]]

 [[1 0 0]
  [0 1 0]]

 [[1 0 0]
  [0 1 0]]]

output tensor shape: (3, 2, 6)

[[[1 0 0 1 0 0]
  [0 1 0 0 1 0]]

 [[1 0 0 1 0 0]
  [0 1 0 0 1 0]]

 [[1 0 0 1 0 0]
  [0 1 0 0 1 0]]]
```

### Causal Attention Function

Now it is time for you to put everything together within the `CausalAttention` or Masked multi-head attention function:

<img src = "masked-attention.png"> 

**Instructions:** Implement the causal attention.
Your model returns the causal attention through a $tl.Serial$ with the following:

- <span style='color:blue'> [tl.Branch](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.combinators.Branch) </span>: consisting of 3 [tl.Dense(d_feature), ComputeAttentionHeads] to account for the queries, keys, and values.
- <span style='color:blue'> [tl.Fn](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.base.Fn)</span>: Takes in dot_product_self_attention function and uses it to compute the dot product using $Q$, $K$, $V$.
- <span style='color:blue'> [tl.Fn](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.base.Fn)</span>: Takes in compute_attention_output_closure to allow for parallel computing.
- <span style='color:blue'> [tl.Dense](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Dense)</span>: Final Dense layer, with dimension `d_feature`.

Remember that in order for trax to properly handle the functions you just defined, they need to be added as layers using the [`tl.Fn()`](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.base.Fn) function. 


```python
# UNQ_C5
# GRADED FUNCTION: CausalAttention
def CausalAttention(d_feature, 
                    n_heads, 
                    compute_attention_heads_closure=compute_attention_heads_closure,
                    dot_product_self_attention=dot_product_self_attention,
                    compute_attention_output_closure=compute_attention_output_closure,
                    mode='train'):
    """Transformer-style multi-headed causal attention.

    Args:
        d_feature (int):  dimensionality of feature embedding.
        n_heads (int): number of attention heads.
        compute_attention_heads_closure (function): Closure around compute_attention heads.
        dot_product_self_attention (function): dot_product_self_attention function. 
        compute_attention_output_closure (function): Closure around compute_attention_output. 
        mode (str): 'train' or 'eval'.

    Returns:
        trax.layers.combinators.Serial: Multi-headed self-attention model.
    """
    
    assert d_feature % n_heads == 0
    d_head = d_feature // n_heads

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # HINT: The second argument to tl.Fn() is an uncalled function (without the parentheses)
    # Since you are dealing with closures you might need to call the outer 
    # function with the correct parameters to get the actual uncalled function.
    ComputeAttentionHeads = tl.Fn('AttnHeads', compute_attention_heads_closure(n_heads, d_head), n_out=1)
        

    return tl.Serial(
        tl.Branch( # creates three towers for one input, takes activations and creates queries keys and values
            [tl.Dense(d_feature), ComputeAttentionHeads], # queries
            [tl.Dense(d_feature), ComputeAttentionHeads], # keys
            [tl.Dense(d_feature), ComputeAttentionHeads], # values
        ),
        
        tl.Fn('DotProductAttn', dot_product_self_attention, n_out=1), # takes QKV
        # HINT: The second argument to tl.Fn() is an uncalled function
        # Since you are dealing with closures you might need to call the outer 
        # function with the correct parameters to get the actual uncalled function.
        tl.Fn('AttnOutput', compute_attention_output_closure(n_heads, d_head), n_out=1), # to allow for parallel
        tl.Dense(d_feature) # Final dense layer
    )

    ### END CODE HERE ###
```


```python
# Take a look at the causal attention model
print(CausalAttention(d_feature=512, n_heads=8))
```

    Serial[
      Branch_out3[
        [Dense_512, AttnHeads]
        [Dense_512, AttnHeads]
        [Dense_512, AttnHeads]
      ]
      DotProductAttn_in3
      AttnOutput
      Dense_512
    ]
    

**Expected Output:**
```CPP
Serial[
  Branch_out3[
    [Dense_512, AttnHeads]
    [Dense_512, AttnHeads]
    [Dense_512, AttnHeads]
  ]
  DotProductAttn_in3
  AttnOutput
  Dense_512
]
```

<a name='2.3'></a>

## 2.3 Transformer decoder block

Now that you have implemented the causal part of the transformer, you will implement the transformer decoder block. Concretely you will be implementing this image now.

<img src = "transformer_decoder_1.png" style = "height:300px"> 

To implement this function, you will have to call the `CausalAttention` or Masked multi-head attention function you implemented above. You will have to add a feedforward which consists of: 

- <span style='color:blue'> [tl.LayerNorm](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.normalization.LayerNorm) </span>: used to layer normalize
- <span style='color:blue'> [tl.Dense](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Dense) </span>: the dense layer
- <span style='color:blue'> [ff_activation](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.activation_fns.Relu) </span>: feed forward activation (we use ReLu) here.
- <span style='color:blue'> [tl.Dropout](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Dropout) </span>: dropout layer
- <span style='color:blue'> [tl.Dense](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Dense) </span>: dense layer
- <span style='color:blue'> [tl.Dropout](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Dropout) </span>: dropout layer

Finally once you implement the feedforward, you can go ahead and implement the entire block using: 

- <span style='color:blue'> [tl.Residual](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.combinators.Residual) </span>: takes in the tl.LayerNorm(), causal attention block, tl.dropout. 

- <span style='color:blue'> [tl.Residual](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.combinators.Residual) </span>: takes in the feedforward block you will implement. 

<a name='ex03'></a>
### Exercise 03
**Instructions:** Implement the transformer decoder block. Good luck!


```python
# UNQ_C6
# GRADED FUNCTION: DecoderBlock
def DecoderBlock(d_model, d_ff, n_heads,
                 dropout, mode, ff_activation):
    """Returns a list of layers that implements a Transformer decoder block.

    The input is an activation tensor.

    Args:
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        mode (str): 'train' or 'eval'.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        list: list of trax.layers.combinators.Serial that maps an activation tensor to an activation tensor.
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # Create masked multi-head attention block using CausalAttention function
    causal_attention = CausalAttention( 
                        d_model,
                        n_heads=n_heads,
                        mode=mode
                        )

    # Create feed-forward block (list) with two dense layers with dropout and input normalized
    feed_forward = [ 
        # Normalize layer inputs
        tl.LayerNorm(),
        # Add first feed forward (dense) layer (don't forget to set the correct value for n_units)
        tl.Dense(d_ff),
        # Add activation function passed in as a parameter (you need to call it!)
        ff_activation(), # Generally ReLU
        # Add dropout with rate and mode specified (i.e., don't use dropout during evaluation)
        tl.Dropout(rate=dropout, mode=mode),
        # Add second feed forward layer (don't forget to set the correct value for n_units)
        tl.Dense(d_model),
        # Add dropout with rate and mode specified (i.e., don't use dropout during evaluation)
        tl.Dropout(rate=dropout,mode=mode)
    ]

    # Add list of two Residual blocks: the attention with normalization and dropout and feed-forward blocks
    return [
      tl.Residual(
          # Normalize layer input
          tl.LayerNorm(),
          # Add causal attention block previously defined (without parentheses)
          causal_attention,
          # Add dropout with rate and mode specified
          tl.Dropout(rate=dropout, mode=mode)
        ),
      tl.Residual(
          # Add feed forward block (without parentheses)
          feed_forward
        ),
      ]
    ### END CODE HERE ###
```


```python
# Take a look at the decoder block
print(DecoderBlock(d_model=512, d_ff=2048, n_heads=8, dropout=0.1, mode='train', ff_activation=tl.Relu))
```

    [Serial[
      Branch_out2[
        None
        Serial[
          LayerNorm
          Serial[
            Branch_out3[
              [Dense_512, AttnHeads]
              [Dense_512, AttnHeads]
              [Dense_512, AttnHeads]
            ]
            DotProductAttn_in3
            AttnOutput
            Dense_512
          ]
          Dropout
        ]
      ]
      Add_in2
    ], Serial[
      Branch_out2[
        None
        Serial[
          LayerNorm
          Dense_2048
          Relu
          Dropout
          Dense_512
          Dropout
        ]
      ]
      Add_in2
    ]]
    

**Expected Output:**
```CPP
[Serial[
  Branch_out2[
    None
    Serial[
      LayerNorm
      Serial[
        Branch_out3[
          [Dense_512, AttnHeads]
          [Dense_512, AttnHeads]
          [Dense_512, AttnHeads]
        ]
        DotProductAttn_in3
        AttnOutput
        Dense_512
      ]
      Dropout
    ]
  ]
  Add_in2
], Serial[
  Branch_out2[
    None
    Serial[
      LayerNorm
      Dense_2048
      Relu
      Dropout
      Dense_512
      Dropout
    ]
  ]
  Add_in2
]]
```

<a name='2.4'></a>
## 2.4 Transformer Language Model

You will now bring it all together. In this part you will use all the subcomponents you previously built to make the final model. Concretely, here is the image you will be implementing. 
<img src = "transformer_decoder.png" style = "height:400px">

    
<a name='ex04'></a>
### Exercise 04
**Instructions:** Previously you coded the decoder block. Now you will code the transformer language model. Here is what you will need. 

- <span style="color:blue"> positional_enconder </span>- a list containing the following layers:
    - <span style="color:blue"> [tl.Embedding](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Embedding)
    - <span style="color:blue"> [tl.Dropout](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Dropout)
    - <span style="color:blue"> [tl.PositionalEncoding](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.attention.PositionalEncoding)

- A list of `n_layers` <span style="color:blue"> decoder blocks</span>.
- <span style="color:blue"> [tl.Serial](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.combinators.Serial): </span> takes in the following layers or lists of layers:
    - <span style="color:blue"> [tl.ShiftRight](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.attention.ShiftRight): </span>: shift the tensor to the right by padding on axis 1.
    - <span style="color:blue"> positional_encoder </span>: encodes the text positions.
    - <span style="color:blue"> decoder_blocks </span>: the ones you created.
    - <span style="color:blue"> [tl.LayerNorm](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.normalization.LayerNorm) </span>: a layer norm.
    - <span style="color:blue"> [tl.Dense](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Dense) </span>: takes in the vocab_size.
    - <span style="color:blue"> [tl.LogSoftmax](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.LogSoftmax) </span>: to predict.
    
Go go go!! You can do it :)




```python
# UNQ_C7
# GRADED FUNCTION: TransformerLM
def TransformerLM(vocab_size=33300,
                  d_model=512,
                  d_ff=2048,
                  n_layers=6,
                  n_heads=8,
                  dropout=0.1,
                  max_len=4096,
                  mode='train',
                  ff_activation=tl.Relu):
    """Returns a Transformer language model.

    The input to the model is a tensor of tokens. (This model uses only the
    decoder part of the overall Transformer.)

    Args:
        vocab_size (int): vocab size.
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_layers (int): number of decoder layers.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        max_len (int): maximum symbol length for positional encoding.
        mode (str): 'train', 'eval' or 'predict', predict mode is for fast inference.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        trax.layers.combinators.Serial: A Transformer language model as a layer that maps from a tensor of tokens
        to activations over a vocab set.
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # Embedding inputs and positional encoder
    positional_encoder = [ 
        # Add embedding layer of dimension (vocab_size, d_model)
        tl.Embedding(vocab_size, d_model),
        # Use dropout with rate and mode specified
        tl.Dropout(rate=dropout, mode=mode),
        # Add positional encoding layer with maximum input length and mode specified
        tl.PositionalEncoding(max_len=max_len, mode=mode)]

    # Create stack (list) of decoder blocks with n_layers with necessary parameters
    decoder_blocks = [ 
        DecoderBlock(d_model, d_ff, n_heads, dropout, mode, ff_activation) for _ in range(n_layers)]

    # Create the complete model as written in the figure
    return tl.Serial(
        # Use teacher forcing (feed output of previous step to current step)
        tl.ShiftRight(mode=mode), # Specify the mode!
        # Add positional encoder
        positional_encoder,
        # Add decoder blocks
        decoder_blocks,
        # Normalize layer
        tl.LayerNorm(),

        # Add dense layer of vocab_size (since need to select a word to translate to)
        # (a.k.a., logits layer. Note: activation already set by ff_activation)
        tl.Dense(vocab_size),
        # Get probabilities with Logsoftmax
        tl.LogSoftmax()
    )

    ### END CODE HERE ###
```


```python
# Take a look at the Transformer
print(TransformerLM(n_layers=1))
```

    Serial[
      ShiftRight(1)
      Embedding_33300_512
      Dropout
      PositionalEncoding
      Serial[
        Branch_out2[
          None
          Serial[
            LayerNorm
            Serial[
              Branch_out3[
                [Dense_512, AttnHeads]
                [Dense_512, AttnHeads]
                [Dense_512, AttnHeads]
              ]
              DotProductAttn_in3
              AttnOutput
              Dense_512
            ]
            Dropout
          ]
        ]
        Add_in2
      ]
      Serial[
        Branch_out2[
          None
          Serial[
            LayerNorm
            Dense_2048
            Relu
            Dropout
            Dense_512
            Dropout
          ]
        ]
        Add_in2
      ]
      LayerNorm
      Dense_33300
      LogSoftmax
    ]
    

**Expected Output:**
```CPP
Serial[
  ShiftRight(1)
  Embedding_33300_512
  Dropout
  PositionalEncoding
  Serial[
    Branch_out2[
      None
      Serial[
        LayerNorm
        Serial[
          Branch_out3[
            [Dense_512, AttnHeads]
            [Dense_512, AttnHeads]
            [Dense_512, AttnHeads]
          ]
          DotProductAttn_in3
          AttnOutput
          Dense_512
        ]
        Dropout
      ]
    ]
    Add_in2
  ]
  Serial[
    Branch_out2[
      None
      Serial[
        LayerNorm
        Dense_2048
        Relu
        Dropout
        Dense_512
        Dropout
      ]
    ]
    Add_in2
  ]
  LayerNorm
  Dense_33300
  LogSoftmax
]
```

<a name='3'></a>
# Part 3: Training

Now you are going to train your model. As usual, you have to define the cost function, the optimizer, and decide whether you will be training it on a `gpu` or `cpu`. In this case, you will train your model on a cpu for a few steps and we will load in a pre-trained model that you can use to predict with your own words.

<a name='3.1'></a>
### 3.1 Training the model

You will now write a function that takes in your model and trains it. To train your model you have to decide how many times you want to iterate over the entire data set. Each iteration is defined as an `epoch`. For each epoch, you have to go over all the data, using your training iterator.

<a name='ex05'></a>
### Exercise 05
**Instructions:** Implement the `train_model` program below to train the neural network above. Here is a list of things you should do:

- Create the train task by calling [`trax.supervised.training.TrainTask`](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.training.TrainTask) and pass in the following: 
    - <span style='color:blue'> labeled_data </span> = train_gen
    - <span style='color:blue'> loss_fn </span> = [tl.CrossEntropyLoss()](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.metrics.CrossEntropyLoss)
    - <span style='color:blue'> optimizer </span> = [trax.optimizers.Adam(0.01)](https://trax-ml.readthedocs.io/en/latest/trax.optimizers.html#trax.optimizers.adam.Adam)
    - <span style='color:blue'> lr_schedule </span> = [lr_schedule](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.lr_schedules.warmup_and_rsqrt_decay)


- Create the eval task by calling [`trax.supervised.training.EvalTask`](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.training.EvalTask) and pass in the following: 
    - <span style='color:blue'> labeled_data </span> = eval_gen
    - <span style='color:blue'> metrics </span> = tl.CrossEntropyLoss() and [tl.Accuracy()](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.metrics.Accuracy)
    
    
- Create the training loop by calling [`trax.supervised.Training.Loop`](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.training.Loop) and pass in the following: 
    - <span style='color:blue'> TransformerLM </span> 
    - <span style='color:blue'> train_task </span> 
    - <span style='color:blue'> eval_task </span> = [eval_task]
    - <span style='color:blue'> output_dir</span> = output_dir
    
You will be using a cross entropy loss, with Adam optimizer. Please read the [Trax](https://trax-ml.readthedocs.io/en/latest/index.html) documentation to get a full understanding. 

The training loop that this function returns can be runned using the `run()` method by passing in the desired number of steps.


```python
from trax.supervised import training

# UNQ_C8
# GRADED FUNCTION: train_model
def training_loop(TransformerLM, train_gen, eval_gen, output_dir = "~/model"):
    '''
    Input:
        TransformerLM (trax.layers.combinators.Serial): The model you are building.
        train_gen (generator): Training stream of data.
        eval_gen (generator): Evaluation stream of data.
        output_dir (str): folder to save your file.
        
    Returns:
        trax.supervised.training.Loop: Training loop.
    '''
    output_dir = os.path.expanduser(output_dir)  # trainer is an object
    lr_schedule = trax.lr.warmup_and_rsqrt_decay(n_warmup_steps=1000, max_value=0.01)

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    train_task = training.TrainTask( 
      labeled_data=train_gen, # The training generator
      loss_layer=tl.CrossEntropyLoss(), # Loss function 
      optimizer=trax.optimizers.Adam(0.01), # Optimizer (Don't forget to set LR to 0.01)
      lr_schedule=lr_schedule,
      n_steps_per_checkpoint=10
    )

    eval_task = training.EvalTask( 
      labeled_data=eval_gen, # The evaluation generator
      metrics=[tl.CrossEntropyLoss(), tl.Accuracy()] # CrossEntropyLoss and Accuracy
    )

    ### END CODE HERE ###

    loop = training.Loop(TransformerLM(d_model=4,
                                       d_ff=16,
                                       n_layers=1,
                                       n_heads=2,
                                       mode='train'),
                         train_task,
                         eval_tasks=[eval_task],
                         output_dir=output_dir)
    
    return loop
```

Notice that the model will be trained for only 10 steps. 

Even with this constraint the model with the original default arguments took a very long time to finish. Because of this some parameters are changed when defining the model that is fed into the training loop in the function above.


```python
# Should take around 1.5 minutes
!rm -f ~/model/model.pkl.gz
loop = training_loop(TransformerLM, train_batch_stream, eval_batch_stream)
loop.run(10)
```

    
    Step      1: Ran 1 train steps in 8.65 secs
    Step      1: train CrossEntropyLoss |  10.41287422
    Step      1: eval  CrossEntropyLoss |  10.41270351
    Step      1: eval          Accuracy |  0.00000000
    
    Step     10: Ran 9 train steps in 61.40 secs
    Step     10: train CrossEntropyLoss |  10.41241550
    Step     10: eval  CrossEntropyLoss |  10.41329861
    Step     10: eval          Accuracy |  0.00000000
    

 <a name='4'></a>
 # Part 4:  Evaluation  

<a name='4.1'></a>
### 4.1 Loading in a trained model

In this part you will evaluate by loading in an almost exact version of the model you coded, but we trained it for you to save you time. Please run the cell below to load in the model.

As you may have already noticed the model that you trained and the pretrained model share the same overall architecture but they have different values for some of the parameters:

    
   `Original (pretrained) model: `                                 
                                       
    TransformerLM(vocab_size=33300, d_model=512, d_ff=2048, n_layers=6, n_heads=8, 
                   dropout=0.1, max_len=4096, ff_activation=tl.Relu)
                   
   `Your model:`
   
    TransformerLM(d_model=4, d_ff=16, n_layers=1, n_heads=2)
   
   **Only the parameters shown for your model were changed. The others stayed the same.**


```python
# Get the model architecture
model = TransformerLM(mode='eval')

# Load the pre-trained weights
model.init_from_file('model.pkl.gz', weights_only=True)
```

<a name='5'></a>
# Part 5: Testing with your own input

You will now test your input. You are going to implement greedy decoding. This consists of two functions. The first one allows you to identify the next symbol. It gets the argmax of the output of your model and then returns that index. 

<a name='ex06'></a>
### Exercise 06
**Instructions:** Implement the next symbol function that takes in the cur_output_tokens and the trained model to return the index of the next word. 


```python
# UNQ_C9
def next_symbol(cur_output_tokens, model):
    """Returns the next symbol for a given sentence.

    Args:
        cur_output_tokens (list): tokenized sentence with EOS and PAD tokens at the end.
        model (trax.layers.combinators.Serial): The transformer model.

    Returns:
        int: tokenized symbol.
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # current output tokens length
    token_length = len(cur_output_tokens)
    # calculate the minimum power of 2 big enough to store token_length
    # HINT: use np.ceil() and np.log2()
    # add 1 to token_length so np.log2() doesn't receive 0 when token_length is 0
    padded_length = 2**int(np.ceil(np.log2(token_length + 1)))

    # Fill cur_output_tokens with 0's until it reaches padded_length
    padded = cur_output_tokens + [0] * (padded_length - token_length)
    padded_with_batch = np.array(padded)[None, :] # Don't replace this 'None'! This is a way of setting the batch dim

    # model expects a tuple containing two padded tensors (with batch)
    output, _ = model((padded_with_batch, padded_with_batch)) 
    # HINT: output has shape (1, padded_length, vocab_size)
    # To get log_probs you need to index output with 0 in the first dim
    # token_length in the second dim and all of the entries for the last dim.
    log_probs = output[0, token_length, :]
    
    ### END CODE HERE ###
    
    return int(np.argmax(log_probs))
```


```python
# Test it out!
sentence_test_nxt_symbl = "I want to fly in the sky."
detokenize([next_symbol(tokenize(sentence_test_nxt_symbl)+[0], model)])
```




    'The'



**Expected Output:**
```CPP
'The'
```

<a name='5.1'></a>
### 5.1 Greedy decoding

Now you will implement the greedy_decode algorithm that will call the `next_symbol` function. It takes in the input_sentence, the trained model and returns the decoded sentence. 

<a name='ex07'></a>
### Exercise 07

**Instructions**: Implement the greedy_decode algorithm. 


```python
# UNQ_C10
# Decoding functions.
def greedy_decode(input_sentence, model):
    """Greedy decode function.

    Args:
        input_sentence (string): a sentence or article.
        model (trax.layers.combinators.Serial): Transformer model.

    Returns:
        string: summary of the input.
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # Use tokenize()
    cur_output_tokens = tokenize(input_sentence) + [0]
    generated_output = [] 
    cur_output = 0 
    EOS = 1 
    
    while cur_output != EOS:
        # Get next symbol
        cur_output = next_symbol(cur_output_tokens, model)
        # Append next symbol to original sentence
        cur_output_tokens.append(cur_output)
        # Append next symbol to generated sentence
        generated_output.append(cur_output)
        print(detokenize(generated_output))
    
    ### END CODE HERE ###
    
    return detokenize(generated_output)
```


```python
# Test it out on a sentence!
test_sentence = "It was a sunny day when I went to the market to buy some flowers. But I only found roses, not tulips."
print(wrapper.fill(test_sentence), '\n')
print(greedy_decode(test_sentence, model))
```

    It was a sunny day when I went to the market to buy some flowers. But
    I only found roses, not tulips. 
    
    :
    : I
    : I just
    : I just found
    : I just found ros
    : I just found roses
    : I just found roses,
    : I just found roses, not
    : I just found roses, not tu
    : I just found roses, not tulips
    : I just found roses, not tulips
    : I just found roses, not tulips.
    : I just found roses, not tulips.<EOS>
    : I just found roses, not tulips.<EOS>
    

**Expected Output:**
```CPP
:
: I
: I just
: I just found
: I just found ros
: I just found roses
: I just found roses,
: I just found roses, not
: I just found roses, not tu
: I just found roses, not tulips
: I just found roses, not tulips
: I just found roses, not tulips.
: I just found roses, not tulips.<EOS>
: I just found roses, not tulips.<EOS>
```


```python
# Test it out with a whole article!
article = "It’s the posing craze sweeping the U.S. after being brought to fame by skier Lindsey Vonn, soccer star Omar Cummings, baseball player Albert Pujols - and even Republican politician Rick Perry. But now four students at Riverhead High School on Long Island, New York, have been suspended for dropping to a knee and taking up a prayer pose to mimic Denver Broncos quarterback Tim Tebow. Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were all suspended for one day because the ‘Tebowing’ craze was blocking the hallway and presenting a safety hazard to students. Scroll down for video. Banned: Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll (all pictured left) were all suspended for one day by Riverhead High School on Long Island, New York, for their tribute to Broncos quarterback Tim Tebow. Issue: Four of the pupils were suspended for one day because they allegedly did not heed to warnings that the 'Tebowing' craze at the school was blocking the hallway and presenting a safety hazard to students."
print(wrapper.fill(article), '\n')
print(greedy_decode(article, model))
```

    It’s the posing craze sweeping the U.S. after being brought to fame by
    skier Lindsey Vonn, soccer star Omar Cummings, baseball player Albert
    Pujols - and even Republican politician Rick Perry. But now four
    students at Riverhead High School on Long Island, New York, have been
    suspended for dropping to a knee and taking up a prayer pose to mimic
    Denver Broncos quarterback Tim Tebow. Jordan Fulcoly, Wayne Drexel,
    Tyler Carroll and Connor Carroll were all suspended for one day
    because the ‘Tebowing’ craze was blocking the hallway and presenting a
    safety hazard to students. Scroll down for video. Banned: Jordan
    Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll (all pictured
    left) were all suspended for one day by Riverhead High School on Long
    Island, New York, for their tribute to Broncos quarterback Tim Tebow.
    Issue: Four of the pupils were suspended for one day because they
    allegedly did not heed to warnings that the 'Tebowing' craze at the
    school was blocking the hallway and presenting a safety hazard to
    students. 
    
    Jordan
    Jordan Ful
    Jordan Fulcol
    Jordan Fulcoly
    Jordan Fulcoly,
    Jordan Fulcoly, Wayne
    Jordan Fulcoly, Wayne Dre
    Jordan Fulcoly, Wayne Drexe
    Jordan Fulcoly, Wayne Drexel
    Jordan Fulcoly, Wayne Drexel,
    Jordan Fulcoly, Wayne Drexel, Tyler
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day.
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not hee
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warn
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the '
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Te
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebow
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    cra
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocki
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hall
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway and
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway and presenting
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway and presenting a
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway and presenting a safety
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway and presenting a safety hazard
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway and presenting a safety hazard to
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway and presenting a safety hazard to
    students
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway and presenting a safety hazard to
    students.
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway and presenting a safety hazard to
    students.<EOS>
    Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
    suspended for one day. Four students were suspended for one day
    because they allegedly did not heed to warnings that the 'Tebowing'
    craze was blocking the hallway and presenting a safety hazard to
    students.<EOS>
    

**Expected Output:**
```CPP
Jordan
Jordan Ful
Jordan Fulcol
Jordan Fulcoly
Jordan Fulcoly,
Jordan Fulcoly, Wayne
Jordan Fulcoly, Wayne Dre
Jordan Fulcoly, Wayne Drexe
Jordan Fulcoly, Wayne Drexel
Jordan Fulcoly, Wayne Drexel,
.
.
.

Final summary:

Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were
suspended for one day. Four students were suspended for one day
because they allegedly did not heed to warnings that the 'Tebowing'
craze was blocking the hallway and presenting a safety hazard to
students.<EOS>
```

**Congratulations on finishing this week's assignment!** You did a lot of work and now you should have a better understanding of the encoder part of Transformers and how Transformers can be used for text summarization.

**Keep it up!**
