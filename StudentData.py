# 讀取、整理由校務資料庫取得的csv file
import pandas as pd
import numpy as np

# 學生的成績
class StudentScore(object):

    course_name = ["reg_no", "1微積分及演習", "1數位邏輯", "1物理", "1物理實驗", "1計算機概論", "2微積分及演習", "2數位邏輯實習",
                   "2物理", "2物理實驗", "1工程數學", "1機率", "1電子學", "1電子學實習", "1電路學", "2工程數學", "2微處理機", "2電子學", "2電子學實習",
                    "2電路學", "1訊號與系統", "2實務專題(一)", "1實務專題(二)"]  # 22
    course_name_103 = ["reg_no", "1微積分及演練", "1數位邏輯", "1物理", "1物理實驗", "1程式設計與實習", "2微積分及演練", "2數位邏輯實習",
                   "2物理", "2物理實驗", "1工程數學", "1機率", "1電子學", "1電子學實習", "1電路學", "2工程數學", "2微處理機", "2電子學", "2電子學實習","2電路學",
                   "1訊號與系統", "2實務專題(一)", "1實務專題(二)"]  # 22
    dataArrays = []
    reg_no = 0
    index = 0

    def loadData(self, file):
        data = pd.read_csv(file, encoding="big5", header=0)
        return data

    def dataArray(self, studentAmount, data):
        for i in range(studentAmount):
            self.dataArrays.append([])

        for i in range(studentAmount):
            for j in range(len(self.course_name)):
                self.dataArrays[i].append(0)

        self.reg_no = data.iloc[0, 2]
        self.dataArrays[0][0] = self.reg_no

    def splitData(self, data):
        length = len(data)

        for i in range(length):

            if (data.iloc[i, 2] == self.reg_no):
                course = str(data.iloc[i, 1]) + data.iloc[i, 3]
                course_index = self.course_name.index(course)
                if self.dataArrays[self.index][course_index] == 0:
                    self.dataArrays[self.index][course_index] = data.iloc[i, 4]

            else:
                self.reg_no = data.iloc[i, 2]
                self.index += 1
                course = str(data.iloc[i, 1]) + data.iloc[i, 3]
                course_index = self.course_name.index(course)
                self.dataArrays[self.index][0] = self.reg_no
                if self.dataArrays[self.index][course_index] == 0:
                    self.dataArrays[self.index][course_index] = data.iloc[i, 4]

    def saveData(self, file):
        df = pd.DataFrame(self.dataArrays, columns=self.course_name)
        df.reset_index(drop=True)
        df.to_csv(file)

# 學生的基本資料
class StudentInfo(object):

    student_info = ["sex", "enter_name", "birthplace", "identity", "f_education", "f_job_type", "m_education", "m_job_type"]
    studentScore = StudentScore()

    def loadData(self, file):
        data = pd.read_csv(file, encoding="big5", header=0)
        return data

    def saveData(self, file, data):

        df = pd.DataFrame(data, columns=self.studentScore.course_name + self.student_info)
        df.reset_index(drop=True)
        df.to_csv(file)

# 整理學生的基本資料 共43種
class CleanStudentInfo(object):

    def loadData(self, file):
        data = pd.read_csv(file, encoding="big5", header=0)
        return data

    def cleanSex(self, data, index):

        for i in range(len(data)):
            if data[i, index] == 1 :
                data[i, index] = "男"
            else:
                data[i, index] = "女"
        # print(data[:, index])
        print(set(data[:, index]))  # 2
        return data

    def cleanBirthplace(self, data, index):
        north = "臺灣省桃園市新北市三峽區恩主公醫新竹縣湖口鄉桃園市八德區台北＿台灣省苗栗台北市北投區自強街50桃園市桃園區台北省基隆市臺灣省新竹縣新北市板橋區貴興路九台灣新竹新北市永和區桃園區桃園市中華民國臺北市桃園縣大溪鎮台灣省基隆市新北市三重台灣省基隆市新竹市經國路一段706新北市板橋區桃園縣龜山鄉新莊區臺灣省新北市苗栗縣竹南鎮台灣省新北市台灣新北市板橋新北市新店市寶中路45台灣桃園台北市新北市土城區臺灣省基隆市台灣省桃園縣桃園市新竹市新竹縣桃園縣新北市台灣省台北市台北縣臺北市竹北苗栗台北縣三重市台灣省台北縣臺北縣臺灣桃園新竹市武陵路271巷91八里苗栗縣臺灣省臺北縣蘆洲"
        central = "臺灣臺中台灣台中台中豐原台中市林森路232號西台灣省嘉義縣台中南投縣草屯鎮台灣省南投市台中市梧棲區台灣彰化縣台灣彰化台灣省彰化線永靖鄉同彰化縣 員林鎮彰化縣鹿港鎮臺灣省嘉義縣南投市台中市霧峰區台中市清水區南投竹山台灣雲林台灣省雲林縣臺灣省嘉義市南投縣埔里鎮同聲裡東台中市豐原區台中大甲台灣省南投縣台中市龍井區田中里舊臺灣省臺中縣台灣省台中市龍井區台中縣南投縣彰化縣嘉義縣民雄鄉富義新村臺灣省臺中市台灣省彰化縣台灣省嘉義市臺中市龍井區彰化縣員林鎮台灣省台中縣台灣省彰化縣嘉義市臺灣省彰化縣彰化市雲林台中縣太平市台中縣豐原市雲林縣斗南鎮田頭里"
        east = "台灣省花蓮縣台灣省宜蘭縣臺灣省花蓮縣台灣花蓮宜蘭縣宜蘭市花蓮縣花蓮市台東縣台東市台灣宜蘭"
        south = "臺灣省高雄縣新化台灣省高雄市台南市永康區永勝街10臺南市台灣省台南縣臺南縣永康市高雄縣高雄市屏東縣屏東市台南縣台南市臺灣省台南市台南縣仁德鄉二行村行台灣省台南市"
        island = "澎湖金門馬祖"
        for i in range(len(data)):
            if str(data[i, index]) in north:
                data[i, index] = "北部"
            elif str(data[i, index]) in central:
                data[i, index] = "中部"
            elif str(data[i, index]) in south:
                data[i, index] = "南部"
            elif str(data[i, index]) in east:
                data[i, index] = "東部"
            # elif data[i, index] in island:
            #     data[i, index] = "外島"
            else:
                data[i, index] = "其他"

        print(data[:, index])
        print(set(data[:, index]), "5") #5
        return data

    def cleanEnterName(self, data, index):

        other = ["海外僑生聯招", "外籍生申請入學", "身心障礙甄試", "陸生聯招", "申請入學"]
        for i in range(len(data)):
            if data[i, index] in other:
                data[i, index] = "其他"

        # print(data[:, index])
        print(set(data[:, index]), "8")  # 8
        return data

    def cleanIdentity(self, data, index):

        other = ["原住民", "領有殘障手冊之視障、聽障、語言障礙及多重障礙學生", "僑生","大陸來台生", "離島生"]
        for i in range(len(data)):
            if data[i, index] in other:
                data[i, index] = "其他"

        # print(data[:, index])
        print(set(data[:, index]), "3")  # 3
        return data

    def cleanEducation(self, data, index):

        for i in range(len(data)):
            if data[i, index] in ["初中(職)"]:
                data[i, index] = "初中職"
            elif data[i, index] in ["高中(職)"]:
                data[i, index] = "高中職"
            elif data[i, index] in ["博士", "碩士"]:
                data[i, index] = "碩博士"
            elif data[i, index] in ["小學", np.nan, "識字(未就學)", "不識字"]:
                data[i, index] = "其他"
        # print(data[:, index])
        print(set(data[:, index]), "6")  # 6
        return data

    def cleanJob(self, data, index):

        for i in range(len(data)):
            if data[i, index] in ["醫護"]:
                data[i, index] = "服務業"
            elif data[i, index] in ["教"]:
                data[i, index] = "公"
            elif data[i, index] in ["農", np.nan]:
                data[i, index] = "其他"
        # print(data[:, index])

            if index == 29:
                if data[i, index] =="家管":
                    data[i, index] = "其他"

        print(set(data[:, index]), "6 or 7")  # 7
        return data


