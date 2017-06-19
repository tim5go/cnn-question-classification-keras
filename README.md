# Recurrent Convolutional Neural Networks for Chinese Question Classification

## Dataset
The data set consists of 1216 pairs of question and question label.
There're 9 question types in total, namely:
0.  NUMBER
1.  PERSON
2.  LOCATION
3.  ORGANIZATION
4.  ARTIFACT
5.  TIME
6.  PROCEDURE
7.  AFFIRMATION
8.  CAUSALITY

## Embedding Preparation
In my experiment, I built a word2vec model on 全网新闻数据(SogouCA) [Sogou Labs] (http://www.sogou.com/labs/resource/ca.php)
You may refer to [word2vec 中文](http://city.shaform.com/blog/2014/11/04/word2vec.html) for the details.
Remember to convert your corpus from simplified Chinese to traditional Chinese.

## Result

Train Loss | Train Accuracy | Validation Loss| Validation Accuracy 
--- | --- | --- | --- 
0.7000 | 87.11% | 0.8945 | 77.87%
 


