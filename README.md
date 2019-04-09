# Isolation Forest Implementation
The goal of this project is to implement the original [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) algorithm by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou.      

**Methodology**     
- Within a random tree, by splitting at random variables and values, anomalies are isolated closer to the root of the tree; whereas normal points are isolated at the deeper end of the tree. And if we have a forest of isolation trees, you can average the height of each instance to make your algorithm more robust. 


**Key Ideas of the Algorithm**       
 - Most existing model-based approaches to anomaly detection construct a profile of *normal instances*, then identify instances that do not conform to the normal profile as anomalies.      
 - Anomalies are ‘few and different’, which make them more susceptible to isolation than normal points.        
 - If you have a binary tree, anomalies are easy to be separated by choosing random features and using uniform random cut. Anomalies are isolated closer to the root of the tree; whereas normal points are isolated at the deeper end of the tree.     
 


