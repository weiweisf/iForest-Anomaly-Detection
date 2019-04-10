# Isolation Forest Implementation
The goal of this project is to implement the original [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) algorithm by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou.      

**Methodology**     
- For a given dataset, we build a random tree on them by splitting at random variables and values. Anomalies are isolated closer to the root of the tree; whereas normal points are isolated at the deeper end of the tree. And if we have a forest of isolation trees, you can average the height of each instance to make your algorithm more robust. 


**Key Ideas of the Algorithm**       
 - Most existing model-based approaches to anomaly detection construct a profile of *normal instances*, then identify instances that do not conform to the normal profile as anomalies.      
 - Anomalies are ‘few and different’, which make them more susceptible to isolation than normal points.        
 - If you have a binary tree, anomalies are easy to be separated by choosing random features and using uniform random cut. Anomalies are isolated closer to the root of the tree; whereas normal points are isolated at the deeper end of the tree.     
 

**Characteristics**    
- Randomly select samples. So the algorithm doesn’t have to walk through all the normal points. Thus more efficient than the existing models that need to build the profiles of normal points first.      
- If data size is large, the trees will grow wider and deeper, the anomalies will not be that distinct from normal points anymore.      


## Data sets

For this project, we'll use three data sets:

* [Kaggle credit card fraud competition data set](https://www.kaggle.com/mlg-ulb/creditcardfraud)    

* Get cancer data into `cancer.csv` by executing [savecancer.csv](https://github.com/parrt/msds689/blob/master/projects/iforest/savecancer.py)     

* [http.zip](https://github.com/parrt/msds689/blob/master/projects/iforest/http.zip)       


