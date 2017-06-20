# Recurrent Convolutional Neural Networks for Chinese Question Classification on BQuLD

## Architecture Overview
![Alt text](https://raw.githubusercontent.com/tim5go/cnn-question-classification-keras/master/img/rcnn_p1.png)
 
For more details  [Click Here](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745).

## Bilingual Question Labelling Dataset (BQuLD)
This dataset is a bilingual (traditional Chinese & English) question labelling dataset designed for NLP researchers. <br />
It originally consists of 1216 pairs of question and question label, which first published by the author of this GitHub [tim5go](https://github.com/tim5go)  <br />
There are 9 question types in total, namely:  <br />

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
In my experiment, I built a word2vec model on 全網新聞數據(SogouCA) [Sogou Labs](http://www.sogou.com/labs/resource/ca.php)  <br />

For example,

```
$ cat news_tensite_xml.dat | iconv -f gbk -t utf-8 -c | grep "<content>" | sed 's\<content>\\' | sed 's\</content>\\' > corpus.txt
$ cws_cmdline --threads 4 --input corpus.txt --segmentor-model cws.model > corpus.seg.txt
$ opencc -i corpus.seg.txt -o corpus_trad.txt
$ nohup ./word2vec -train corpus_trad.txt -output sogou_vectors.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1 &
```

You may refer to [word2vec 中文](http://city.shaform.com/blog/2014/11/04/word2vec.html) for the details.  <br />
Remember to convert your corpus from simplified Chinese to traditional Chinese.  <br />

## Result

Training Loss | Training Accuracy | Validation Loss| Validation Accuracy 
--- | --- | --- | --- 
0.7000 | 87.11% | 0.8945 | 77.87%
 


