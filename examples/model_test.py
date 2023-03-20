import pandas as pd
from parse_dataset import preprocess
from model_train import model

# 训练模型加载
model = model()
test_file = "../data/adult/adult.test"
# 测试集预处理，去掉需要进行预测的特征
test_data = preprocess(test_file, 0).drop(['earn'], axis=1)


def validate():
    # 测试集已给结果读取
    ground_truth = pd.read_csv('res1.data')
    ground_truth['earn'] = ground_truth['earn'].apply(lambda x: x.split('.')[0].strip())

    # 模型预测
    predict = model.predict(test_data, n_jobs=4)

    # 预测结果处理
    predict['earn'] = predict['earn'].apply(lambda x: x.strip())
    print("predict results:\n", predict)
    print("predict accurate rate: ",  (predict['earn'] == ground_truth['earn']).sum() / len(ground_truth))


if __name__ == '__main__':
    validate()
