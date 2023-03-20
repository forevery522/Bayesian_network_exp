import pandas as pd
import numpy as np

# adult数据集标签名， 特征共15个
names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
         "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
         "native-country", "earn"]


# 对adult数据集进行读取
def read_dataset(filename):
    dataset = pd.read_csv(filename, names=names)
    return dataset


# 数据预处理
def preprocess(file, sample_percent):
    data = read_dataset(file)

    # 删除无关特征
    data.drop(columns="fnlwgt", inplace=True)
    data.drop(columns="education-num", inplace=True)

    # 连续变量特征转换为离散型特征值，便于贝叶斯网络参数学习
    data.loc[data['age'] <= 20, 'age'] = 0
    data.loc[(data['age'] > 20) & (data['age'] <= 60), 'age'] = 1
    data.loc[(data['age'] > 60), 'age'] = 2

    # 合并处理相似特征
    data["capital-remain"] = data["capital-gain"] - data["capital-loss"]

    data.loc[(data["capital-remain"] <= 5000), "capital-remain"] = 0
    data.loc[(data["capital-remain"] > 5000), "capital-remain"] = 1

    data.loc[(data["hours-per-week"] <= 49), "hours-per-week"] = 0
    data.loc[(data["hours-per-week"] > 49), "hours-per-week"] = 1

    # 合并产生新特征后，删除冗余特征
    data.drop(columns="capital-gain", inplace=True)
    data.drop(columns="capital-loss", inplace=True)

    data[data.columns] = data[data.columns].astype(str)

    # 数据集的缺失采样， 缺失值采样由缺失率决定
    data = sampleUnKnown(data, sample_percent)

    # 数据缺失值处理，以NaN代替缺失值
    for col in data.columns:
        if data[col].dtype != np.int64:
            data.loc[data[col].str.contains('\?', na=False), col] = "NaN"

    print("\n" + file.split('.')[3] + ": ")
    print("UnKnown data sample rate: \n", unKnownCount(data))

    return data


def sampleUnKnown(data, percent):
    # 排除需要预测的特征值，避免预测偏差
    for col in data.columns:
        if col == "earn":
            continue
        data.loc[data[col].sample(frac=percent).index, col] = "NaN"
    return data


# 各特征缺失率统计
def unKnownCount(data):
    return data.isin(["NaN"]).sum() / data.shape[0]
