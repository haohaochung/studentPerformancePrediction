import numpy as np
from StudentData import StudentInfo, StudentScore, CleanStudentInfo
import seaborn as sns
import matplotlib.pyplot as plt


studentScore = StudentScore()
studentInfo = StudentInfo()
cleanStudentInfo = CleanStudentInfo()
year = "101"

# 1. 抓學生的成績
originalDataScore = studentScore.loadData("studentData/"+year+"_trainingdata.csv")
studentScore.dataArray(150, originalDataScore)
studentScore.splitData(originalDataScore)
studentScore.saveData("studentData/studentInfo/"+year+"_trainingData.csv")

# 2. 抓學生的基本資料  & combine
dataInfo = studentInfo.loadData("studentData/studentInfo/"+year+"_studentInfo.csv")
dataScore = studentScore.loadData("studentData/studentInfo/"+year+"_trainingData.csv")
merged = dataScore.merge(dataInfo, on="reg_no")
studentInfo.saveData("studentData/studentInfo/"+year+"_cleanData.csv", merged)

# # 3. 整理merge的資料
data = cleanStudentInfo.loadData("studentData/studentInfo/"+year+"_cleanData.csv")
data = np.array(data)
data = cleanStudentInfo.cleanSex(data, 24) #入學方式
data = cleanStudentInfo.cleanEnterName(data, 25) #入學方式
data = cleanStudentInfo.cleanBirthplace(data, 26) #出身地
data = cleanStudentInfo.cleanIdentity(data, 27) #身分
data = cleanStudentInfo.cleanEducation(data, 28) #父親教育程度
data = cleanStudentInfo.cleanJob(data, 29) #父親工作
data = cleanStudentInfo.cleanEducation(data, 30) #母親教育程度
data = cleanStudentInfo.cleanJob(data, 31) #母親工作
studentInfo.saveData("studentData/studentInfo/"+year+"_cleanData.csv", data[:,1:])


## ----------------- seaborn ---------------------
# data = cleanStudentInfo.loadData("studentData/studentInfo/"+year+"_cleanData.csv")
#
# sns.countplot(data.iloc[:,24])
# plt.show()
