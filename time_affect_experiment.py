import pandas as pd
import numpy as np

def loadData(file):
    data = pd.read_csv(file, encoding="big5", header=0)
    return data

def dataArray(studentAmount, data):
    for i in range(studentAmount):
        dataArrays.append([])

    for i in range(studentAmount):
        for j in range(len(course_name_103)):
            dataArrays[i].append(0)

    reg_no = data.iloc[0, 1]
    sex = data.iloc[0, 2]
    class_name = data.iloc[0, 0]
    dataArrays[0][0] = reg_no
    dataArrays[0][1] = sex
    dataArrays[0][2] = class_name

def splitData(data):
    length = len(data)
    index = 0
    reg_no = 0
    for i in range(length):

        if (data.iloc[i, 1] == reg_no):
            course = str(data.iloc[i, 4]) + data.iloc[i, 5]
            course_index = course_name_103.index(course)
            if  dataArrays[index][course_index] == 0:
                dataArrays[index][course_index] = str(data.iloc[i, 6])
            elif dataArrays[index][course_index] !=0:
                dataArrays[index][course_index] += ","+ str(data.iloc[i, 6])

        else:
            reg_no = data.iloc[i, 1]
            sex = data.iloc[i, 2]
            class_name = data.iloc[i, 0]
            index += 1
            course = str(data.iloc[i, 4]) + data.iloc[i, 5]
            course_index = course_name_103.index(course)
            dataArrays[index][0] = reg_no
            dataArrays[index][1] = sex
            dataArrays[index][2] = class_name
            if dataArrays[index][course_index] == 0:
                dataArrays[index][course_index] = str(data.iloc[i, 6])
            elif dataArrays[index][course_index] != 0:
                dataArrays[index][course_index] += "," + str(data.iloc[i, 6])

def saveData(file, data):
    df = pd.DataFrame(data, columns=course_name_103)
    df.reset_index(drop=True)
    df.to_csv(file)

def course(num):
    train_10 = [1, 2, 3, 5, 6, 8]
    # train_10 = [1,2,3,4,5,6,7,8,9,10,11,12]
    train_11 = [1, 2, 3, 5, 6, 8]  # 微積分及演習....2電路學
    train_12 = [1, 2, 3, 5, 6, 8, 10, 11]
    train_13 = [4, 7, 9]
    train_14 = [1, 2, 3, 5, 6, 8, 10, 11, 12]
    train_15 = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14]
    train_16 = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14, 15]
    train_17 = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14, 15, 16]
    train_18 = [4, 7, 9, 13]
    train_19 = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14, 15, 16, 17]
    train_20 = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14, 15, 16, 17, 19]
    train_21 = [4, 7, 9, 13, 18]
    train_22 = [4, 7, 9, 13, 18, 21]

    train = [train_10, train_11, train_12, train_13, train_14, train_15, train_16, train_17, train_18, train_19,
             train_20, train_21, train_22]
    predict = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    return train[num], predict[num]

course_name_103 = ["reg_no", "sex","class","1微積分及演練", "1數位邏輯", "1物理", "1物理實驗", "1程式設計與實習", "2微積分及演練", "2數位邏輯實習",
                   "2物理", "2物理實驗", "1工程數學", "1機率", "1電子學", "1電子學實習", "1電路學", "2工程數學", "2微處理機", "2電子學", "2電子學實習", "2電路學",
                   "1訊號與系統", "2實務專題(一)", "1實務專題(二)"]  # 22

dataArrays = []
year = 105

# 0 全早一天 1 全下午一天 2 早多123 3早多34  4下午多567  5下午多789,  6全早多12  7權早多34  8全下午多56 9全下午多78  10其他

# 0:全早 1:全下午 2:早多 3:晚多

# 1. 數位邏輯1
# dict = {"100":[0,1,2], "101":[1,1,1], "102":[3,3,3], "103":[1,3,3], "104":[1,1,1], "105":[1,1,1], "106":[2,1,2]}

# 3. 物理1
# dict = {"100":[3,0,0], "101":[3,3,0], "102":[2,0,1], "103":[1,0,2], "104":[1,3,1], "105":[3,3,1], "106":[1,1,1]}

# 4. 物理實驗1
# dict = {"100":[0,1,1], "101":[0,1,1], "102":[1,1,0], "103":[1,1,1], "104":[0,1,1], "105":[0,1,1], "106":[1,1,1]}

# 5. 程式設計1
# dict = {"103":[1,1,1], "104":[1,0,1], "105":[1,0,1], "106":[1,0,1]}

# 10. 工程數學1
# dict = {"101":[1,1,0], "102":[0,0,1], "103":[2,1,0], "104":[3,1,0], "105":[2,1,1], "106":[0,0,0]}

# 11. 機率1
# dict = {"101":[1,1,0], "102":[0,0,1], "103":[2,1,0], "104":[3,1,0], "105":[2,1,1], "106":[0,0,0]}

# 14. 電路學1
# dict = {"101":[2,1,1], "102":[1,3,1], "103":[2,1,3], "104":[0,1,2], "105":[1,2,1], "106":[0,1,2]}
#
# def choose(year, i):
#     if "甲" in data.iloc[i,0]:
#         return dict[str(year)][0]
#     elif "乙" in data.iloc[i,0]:
#         return dict[str(year)][1]
#     elif "丙" in data.iloc[i,0]:
#         return dict[str(year)][2]
#     else:
#         print("no class")
#
#
#
# data = loadData("studentData/experiment/course_14.csv")
# time = []
#
# zeroList = []
# for i in range(len(data)):
#     if data.iloc[i, 6] < 0:
#         zeroList.append(i)
#
# data = data.drop(data.index[zeroList])
# print(data.iloc[:,6])
# for i in range(len(data)):
#     if  str(data.iloc[i,3]) == "100":
#         ans = choose(100, i)
#     elif  str(data.iloc[i,3]) == "101":
#         ans = choose(101, i)
#     elif  str(data.iloc[i,3]) == "102":
#         ans = choose(102, i)
#     elif  str(data.iloc[i,3]) == "103":
#         ans = choose(103, i)
#     elif  str(data.iloc[i,3]) == "104":
#         ans = choose(104, i)
#     elif  str(data.iloc[i,3]) == "105":
#         ans = choose(105, i)
#     elif  str(data.iloc[i,3]) == "106":
#         ans = choose(106, i)
#     else:
#         print("no year"+i)
#
#     time.append(ans)
#
# print((time))
#
# for i in range(len(data)):
#     data.iloc[i, 5] = time[i]
#
# data.to_csv("studentData/experiment/clean_course_14.csv")



# ==================== 參數影響實驗 ========================= #
data = loadData("experiment/time_affect/104_cleanData.csv")
data.dropna(inplace=True)
train, predict = course(0)
##----- 處理負值 -----##
zeroList = []
for i in range(len(data)):
    if (data.iloc[i, predict] <= 0):  # & (data.iloc[i, predict] != -3):
        zeroList.append(i)
    for j in train:
        if (data.iloc[i, j] <= 0):  # & (data.iloc[i, j] != -3):
            zeroList.append(i)
            break
            ##----- 處理負值 -----##

data = data.drop(data.index[zeroList])
print(zeroList, len(data))

for i in range(len(data)):
    if "男" in data.iloc[i, 23]:
        data.iloc[i, 23] = 0
    else:
        data.iloc[i, 23] = 1

    if "繁星" in data.iloc[i, 24]:
        data.iloc[i, 24] = 0
    elif "技優甄審入學" in data.iloc[i, 24]:
        data.iloc[i, 24] = 1
    elif "推薦甄試" in data.iloc[i, 24]:
        data.iloc[i, 24] = 2
    elif "保送甄試" in data.iloc[i, 24]:
        data.iloc[i, 24] = 3
    elif "菁英班入學" in data.iloc[i, 24]:
        data.iloc[i, 24] = 4
    elif "聯合招生" in data.iloc[i, 24]:
        data.iloc[i, 24] = 5
    elif "其他" in data.iloc[i, 24]:
        data.iloc[i, 24] = 6

    if "北部" in data.iloc[i, 25]:
        data.iloc[i, 25] = 0
    elif "中部" in data.iloc[i, 25]:
        data.iloc[i, 25] = 1
    elif "南部" in data.iloc[i, 25]:
        data.iloc[i, 25] = 2
    elif "東部" in data.iloc[i, 25]:
        data.iloc[i, 25] = 3
    elif "其他" in data.iloc[i, 25]:
        data.iloc[i, 25] = 4

    if "一般生" in data.iloc[i, 26]:
        data.iloc[i, 26] = 0
    elif "外籍生" in data.iloc[i, 26]:
        data.iloc[i, 26] = 1
    elif "其他" in data.iloc[i, 26]:
        data.iloc[i, 26] = 2

    if "其他" in data.iloc[i, 27]:
        data.iloc[i, 27] = 0
    elif "初中職" in data.iloc[i, 27]:
        data.iloc[i, 27] = 1
    elif "高中職" in data.iloc[i, 27]:
        data.iloc[i, 27] = 2
    elif "專科" in data.iloc[i, 27]:
        data.iloc[i, 27] = 3
    elif "學士" in data.iloc[i, 27]:
        data.iloc[i, 27] = 4
    elif "碩博士" in data.iloc[i, 27]:
        data.iloc[i, 27] = 5

    if "其他" in data.iloc[i, 29]:
        data.iloc[i, 29] = 0
    elif "初中職" in data.iloc[i, 29]:
        data.iloc[i, 29] = 1
    elif "高中職" in data.iloc[i, 29]:
        data.iloc[i, 29] = 2
    elif "專科" in data.iloc[i, 29]:
        data.iloc[i, 29] = 3
    elif "學士" in data.iloc[i, 29]:
        data.iloc[i, 29] = 4
    elif "碩博士" in data.iloc[i, 29]:
        data.iloc[i, 29] = 5

    if "公" in data.iloc[i, 28]:
        data.iloc[i, 28] = 0
    elif "工" in data.iloc[i, 28]:
        data.iloc[i, 28] = 1
    elif "服務業" in data.iloc[i, 28]:
        data.iloc[i, 28] = 2
    elif "自由業" in data.iloc[i, 28]:
        data.iloc[i, 28] = 3
    elif "商" in data.iloc[i, 28]:
        data.iloc[i, 28] = 4
    elif "其他" in data.iloc[i, 28]:
        data.iloc[i, 28] = 5
    elif "家管" in data.iloc[i, 28]:
        data.iloc[i, 28] = 6

    if "公" in data.iloc[i, 30]:
        data.iloc[i, 30] = 0
    elif "工" in data.iloc[i, 30]:
        data.iloc[i, 30] = 1
    elif "服務業" in data.iloc[i, 30]:
        data.iloc[i, 30] = 2
    elif "自由業" in data.iloc[i, 30]:
        data.iloc[i, 30] = 3
    elif "商" in data.iloc[i, 30]:
        data.iloc[i, 30] = 4
    elif "其他" in data.iloc[i, 30]:
        data.iloc[i, 30] = 5
    elif "家管" in data.iloc[i, 30]:
        data.iloc[i, 30] = 6

# print(data.iloc[:,24])

# list = [24,25,26,27,28,29,30]
# print(list)
# s = []
# index=0
# for i in range(len(list)):
#     s.append([])
#
# for i in list:
#     for j in range(len(data)):
#         s[index].append(data.iloc[j,i])
#     index += 1
#
# s = np.array(s)
# print(s)
# df = pd.DataFrame(s)
# df.transpose()
# df.columns = ["24","25","26","27","28","29","30"]
data.to_csv("experiment/time_affect/new_104_cleanData.csv")


# ===================== 整理資料庫csv 檔  ===================== #

# data = loadData("studentData/experiment/"+str(year)+"_class.csv")
# dataArray(150, data)
# splitData(data)
# saveData("studentData/experiment/"+str(year)+"_clean_class.csv", dataArrays)

# ===================== Split 甲乙丙 班===================== #

# data = loadData("studentData/experiment/"+str(year)+"_clean_class.csv")
# a = []
# b = []
# c = []
# class_ = [a, b, c]
#
# for i in range(len(data)):
#     if "電機二甲" in data.iloc[i,2]:
#         a.append(data.iloc[i,:])
#     elif "電機二乙" in data.iloc[i, 2]:
#         b.append(data.iloc[i, :])
#     elif "電機二丙" in data.iloc[i, 2]:
#         c.append(data.iloc[i, :])
#
# for i in range(len(class_)):
#     saveData("studentData/experiment/"+str(year)+"_clean_"+str(i)+".csv", class_[i])

# ===================== Training ===================== #


def training(year, classes, train_subject):

    data = loadData("studentData/experiment/"+str(year)+"_clean_"+str(classes)+".csv")
    data.dropna(inplace=True)
    train, predict = course()

    predict_score = []
    train_score = []

    for i in range(len(data)):
        train_score.append([])

    zeroList = []
    for i in range(len(data)):
        score = str(data.iloc[i, predict[train_subject] + 2])
        if "," in score:
            a = int(score.split(",")[0])
            b = int(score.split(",")[1])
            if "-10" in score:
                predict_score.append(a)
            elif (a>0) & (b>0) :
                predict_score.append( (a+b) /2)
            else:
                predict_score.append(0)
        else:
            predict_score.append(int(score))

    for i in range(len(data)):
        for j in train[train_subject]:
            score = str(data.iloc[i, j+2])
            if "," in score:
                a = int(score.split(",")[0])
                b = int(score.split(",")[1])
                if "-10" in score:
                    train_score[i].append(a)
                elif (a>0) & (b>0) :
                    train_score[i].append( (a+b) /2)
                else:
                    train_score[i].append(0)
            else:
                train_score[i].append(int(score))

    ##----- 處理負值 -----##

    for i in range(len(train_score)):
        if predict_score[i] <= 0:
            zeroList.append(i)
        for j in range(len(train[train_subject])):
            if train_score[i][j] == -3:
                train_score[i][j] = 40
            elif train_score[i][j] <= 0:
                zeroList.append(i)
                break

    train_score = np.delete(train_score, zeroList, axis=0)
    predict_score = np.delete(predict_score, zeroList, axis=0)
    print(train_score.shape, predict_score.shape)
    # ----- 處理負值 -----##
    return  train_score, predict_score


