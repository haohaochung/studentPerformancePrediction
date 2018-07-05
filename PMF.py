import numpy as np
import pandas as pd
from sklearn import cluster, datasets, metrics
import seaborn as sns
import matplotlib.pyplot as plt

class TransformData(object):

    def loadData(self, data):
        training_data = pd.read_csv(data, encoding="big5", header=0)
        return training_data

    def saveData(self, pmfData):
        df = pd.DataFrame(pmfData)
        df.to_csv('studentData/pmf3.csv')

    def deleteBiasData(self, pmfData):
        # 處理負值資料
        deleteList = []
        for i in range(len(pmfData)):
            for j in range(len(pmfData[0])):
                if pmfData[i, j] <= 0:
                    if j not in deleteList:
                        deleteList.append(j)

        pmfData = np.delete(pmfData, deleteList, axis=1)
        return pmfData


class Cluster(object):

    # 最多分成五類
    zero = []
    one = []
    two = []
    three = []
    four = []

    def loadData(self, data):
        training_data = pd.read_csv(data, encoding="big5", header=None)
        return training_data

    def kMeans(self, data, count):
        kmeans_fit = cluster.KMeans(n_clusters=count).fit(data)
        cluster_labels = kmeans_fit.labels_
        print(cluster_labels)
        return  cluster_labels

    def doCluster(self, label, course_name, count):

        for i in range(len(course_name)):
            if label[i] == 0:
                self.zero.append(course_name[i])
            elif label[i] == 1:
                self.one.append(course_name[i])
            elif label[i] == 2:
                self.two.append(course_name[i])
            elif label[i] == 3:
                self.three.append(course_name[i])
            else:
                self.four.append(course_name[i])

    def printCluster(self):

        params = {"zero": self.zero, "one": self.one, "two": self.two, "three": self.three, "four": self.four}
        print(params)

    def efficiency(self, data):
        avgs =[]
        r = range(2, 8)
        for i in r:
            kmeans_fit = cluster.KMeans(n_clusters=i).fit(data)
            label = kmeans_fit.labels_
            avg = metrics.silhouette_score(data, label)
            avgs.append(avg)
        plt.bar(r, avgs)
        plt.show()
        print(avgs)
#
# 3. 進行分類
c = Cluster()
data = c.loadData("studentData/pmf.csv")
data = np.array(data)

np.random.shuffle(np.transpose(data))


course_name = ["1微積分及演習-1", "1數位邏輯-2", "1物理-3", "1物理實驗-4", "1計算機概論-5", "2微積分及演習-6", "2數位邏輯實習-7",
               "2物理-8", "2物理實驗-9", "1工程數學-10", "1機率-11", "1電子學-12", "1電子學實習-13", "1電路學-14", "2工程數學-15", "2微處理機-16", "2電子學-17", "2電子學實習-18",
               "2電路學-19", "1訊號與系統-20", "2實務專題(一)-21", "1實務專題(二)-22"]  # 22
count = 2
label = c.kMeans(data, count)
c.doCluster(label, course_name, count)
c.printCluster()
c.efficiency(data)



# # # 1. 進行抓資料
# trans = TransformData()
# data = trans.loadData("studentData/studentInfo/100_cleanData.csv")
# data.dropna(inplace=True)
# data = np.array(data)
# length = len(data)
#
# pmfData = []
# index = 0
#
# for i in range(22):
#     pmfData.append([])
#
# for i in range(1, 23):
#     for j in range(0, length):
#         pmfData[index].append(data[j, i])
#     index+=1
#
#
# # 2. 進行刪除
#
# pmfData = np.array(pmfData)
# print(pmfData.shape)
#
# pmfData = trans.deleteBiasData(pmfData)
# print(pmfData.shape)
# trans.saveData(pmfData)

