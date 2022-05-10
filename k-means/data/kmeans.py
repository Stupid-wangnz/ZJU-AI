from cProfile import label
import os
import sklearn
import warnings
import copy
import random
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
%matplotlib inline

file_dir = './data'
csv_files = os.listdir(file_dir)

# df 作为最后输出的 DataFrame 初始化为空
df = pd.DataFrame()
feature = ['cpc', 'cpm']
df_features = []
for col in feature:
    infix = col + '.csv'
    path = os.path.join(file_dir, infix)
    df_feature = pd.read_csv(path)
    # 将两个特征存储起来用于后续连接
    df_features.append(df_feature)

# 2 张 DataFrame 表按时间连接

df = pd.merge(left=df_features[0], right=df_features[1])
# 将 timestamp 列转化为时间类型
df['timestamp'] = pd.to_datetime(df['timestamp'])
# 将 df 数据按时间序列排序，方便数据展示
df = df.sort_values(by='timestamp').reset_index(drop=True)
# 尝试获取时间关系
df['hours'] = df['timestamp'].dt.hour
df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp').reset_index(drop=True)
# 尝试引入非线性关系
df['cpc X cpm'] = df['cpm'] * df['cpc']
df['cpc / cpm'] = df['cpc'] / df['cpm']

from sklearn.decomposition import PCA

#在进行特征变换之前先对各个特征进行标准化
columns = ['hours','daylight','cpc', 'cpm', 'cpc X cpm', 'cpc / cpm']
data = df[columns]
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data, columns=columns)


#通过 n_components 指定需要降低到的维度
n_components = 4
pca = PCA(n_components=n_components)
data = pca.fit_transform(data)
data = pd.DataFrame(data,columns=['Dimension' + str(i+1) for i in range(n_components)])

class KMeans():
    """
    Parameters
    ----------
    n_clusters 指定了需要聚类的个数，这个超参数需要自己调整，会影响聚类的效果
    n_init 指定计算次数，算法并不会运行一遍后就返回结果，而是运行多次后返回最好的一次结果，n_init即指明运行的次数
    max_iter 指定单次运行中最大的迭代次数，超过当前迭代次数即停止运行
    """
    def __init__(
                self,
                n_clusters=8,
                n_init=10,
                max_iter=300
                ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init

    #def init_centers(self,x):
    #Kmeans++初始化簇心方法
    #返回初始化的簇心数组
    def init_centers_plus(self,x):
        #随机选取一个点作为簇心
        center=x[np.random.randint(0,x.shape[0])]
        centers=np.array([center])
        for i in range(self.n_clusters-1):
            #计算每个点到簇心的距离
            distance=np.array([self.dist(x_i,center) for x_i in x])
            #计算每个点到簇心的概率
            probability=distance/distance.sum()
            #随机选取一个点作为新的簇心
            center=x[np.random.choice(x.shape[0],p=probability)]
            centers=np.append(centers,[center],axis=0)
        return centers

    def dist(self,x,y):
        #计算n维向量x、y的欧氏距离
        distance=0
        for i in range(len(x)):
            distance+=(x[i]-y[i])*(x[i]-y[i])
        return distance**0.5

    def predict(self,x,center):
        #计算每个数据属于哪个簇
        x=x.to_numpy()
        return np.array([np.argmin([self.dist(x_i,center_i) for center_i in center]) for x_i in x])
    
    def update(self,x,labels):
        #根据聚类结果更新聚类中心
        centers=[]
        for i in range(self.n_clusters):
            centers.append(np.mean(x[labels == i],axis=0))
        return np.array(centers)   
    
    def calc_inertia(self,x,labels,center):
        #计算总的簇内误差
        return np.sum([np.sum(np.square(x[labels == i] - center_i)) for i,center_i in enumerate(center)])

    def fit(self, x):
        """
        用fit方法对数据进行聚类
        :param x: 输入数据, shape=(Dimension1, Dimension2, ..., DimensionN)
        :best_centers: 簇中心点坐标 数据类型: ndarray
        :best_labels: 聚类标签 数据类型: ndarray
        :return: self
        """
        ###################################################################################
        #### 请勿修改该函数的输入输出 ####
        ###################################################################################
        # #
        # 初始化簇中心点
        centers = np.random.rand(self.n_clusters, x.shape[1])
        # 初始化簇标签
        labels = np.zeros(x.shape[0])
        # 初始化最佳簇中心点
        best_centers = np.zeros(centers.shape)
        # 初始化最佳簇标签
        best_labels = np.zeros(labels.shape)
        # 初始化最佳簇中心点误差
        best_inertia = np.inf
        
        # 运算n_init次
        for _ in range(self.n_init):
            #迭代max_iter次
            # #初始化单次迭代中最佳簇中心点误差
            best_inertia_iter = np.inf
            # 初始化迭代次数
            n_iter = 0
            
            # 初始化簇中心点
            #centers = np.random.rand(self.n_clusters, x.shape[1])
            centers = self.init_centers_plus(x)
            
            # 初始化簇标签
            labels = np.zeros(x.shape[0])
            
            # 初始化迭代中的簇标签、簇中心点
            best_centers_iter = np.zeros(centers.shape)
            best_labels_iter = np.zeros(labels.shape)

            while n_iter < self.max_iter:
                #对数据进行聚类
                labels = self.predict(x, centers)
                #更新质心
                centers = self.update(x, labels)
                #计算簇中心点误差
                inertia = self.calc_inertia(x, labels, centers)
                
                #记录最佳簇中心点
                if inertia < best_inertia_iter:
                    best_centers_iter = centers
                    best_labels_iter = labels
                    best_inertia_iter = inertia
                else:
                    break
                #迭代次数+1
                n_iter += 1
            print(n_iter)
            #记录最佳簇中心点误差
            if best_inertia_iter < best_inertia:
                best_inertia = best_inertia_iter
                best_centers = best_centers_iter
                best_labels = best_labels_iter


        # #
        ###################################################################################
        ############# 在生成 main 文件时, 请勾选该模块 ############# 
        ###################################################################################
        best_centers = best_centers
        best_labels = best_labels
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        return self

    