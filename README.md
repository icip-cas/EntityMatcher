EntityMatcher
=============

EntityMatcher is a Python package including implementations of multiple deep entity matching models proposed by our group. The current version only contains the HierMatcher model proposed in IJCAI-2020 paper ["Hierarchical Matching Network for Heterogeneous Entity Resolution"](https://www.ijcai.org/Proceedings/2020/0507.pdf). More models ([MPM](https://www.ijcai.org/Proceedings/2019/0689.pdf), [Seq2SeqMatcher](https://dl.acm.org/doi/pdf/10.1145/3357384.3358018), ect.) will be available later.

EntityMatcher is built on the framework of [DeepMatcher](https://github.com/anhaidgroup/deepmatcher), which is an easily customizable deep entity matching package.

## Environment Setting
* Python 3.6
* scikit-learn 0.22.2
* deepmatcher 0.1.1

## Datasets
There are ten datasets of three types in the “data/” directory of this project:
1. Four pubic homogeneous datasets, which are originally obtained from [here](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md).  
  Walmart-Amazon<sub>1</sub>: "data/walmart_amazon"  
  Amazon-Google: "data/amazon_google"  
  DBLP-ACM<sub>1</sub>: "data/dblp_acm"  
  DBLP-Scholar<sub>1</sub>: "data/dblp_scholar"  

2. Three public dirty datasets, which are originally obtained from [here](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md).  
  Walmart-Amazon<sub>2</sub>: "data/dirty_walmart_amazon"  
  DBLP-ACM<sub>2</sub>: "data/dirty_dblp_acm"  
  DBLP-Scholar<sub>2</sub>: "data/dirty_dblp_scholar"  

3. Three heterogeneous datasets, which are derived from Walmart-Amazon<sub>1</sub> using different attribute merging operations (see more details from [here](https://www.ijcai.org/Proceedings/2020/0507.pdf)).  
  Walmart-Amazon<sub>3</sub>: "data/walmart_amazon_3"  
  Walmart-Amazon<sub>4</sub>: "data/walmart_amazon_4"  
  Walmart-Amazon<sub>5</sub>: "data/walmart_amazon_5"  
  
All of the above datasets have been processed according to the input data format of [DeepMatcher](https://github.com/anhaidgroup/deepmatcher), thus can be directly used.

  ## Embedding file
Download fastText model file trained on English Wikipedia from [here](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip).  Then unzip it and copy the file named "wiki.en.bin" to the “embedding/” directory of this project.   

  ## Quick start
  Run experiments on specified dataset and model:
  ```
      python run.py -m <model_name>  -d <dataset_dir>  -e <embedding_dir> 
  ```
  For example, to run an experiment on Walmart-Amazon with HierMatcher model, use:
  ```
      python run.py -m "HierMatcher" -d "data/walmart_amazon/" -e "embedding"
  ```
  
 ## Citation
Please cite our work if you like or are using our codes for your projects:   
```
Cheng Fu, Xianpei Han, Jiaming He and Le Sun, Hierarchical Matching Network for Heterogeneous Entity Resolution. IJCAI 2020: 3665-3671
```

 ## The Team
EntityMatcher is developed by [Chinese Information Processing Laboratory (CIP)](http://www.icip.org.cn/zh/homepage/), Institute of Software , Chinese Academy of Science. 
