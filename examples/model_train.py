from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization

from parse_dataset import preprocess

train_file = "../data/adult/adult.data"

# 预处理部分，采样缺失率设置
# train_data = preprocess(train_file, 0)
# train_data = preprocess(train_file, 0.1)
train_data = preprocess(train_file, 0.2)


# 创建贝叶斯网络模型
def create_model():
    model = BayesianNetwork([
        ('age', 'education'),
        ('age', 'earn'),
        ('age', 'capital-remain'),
        ('education', 'occupation'),
        ('marital-status', 'relationship'),
        ('occupation', 'workclass'),
        ('occupation', 'hours-per-week'),
        ('hours-per-week', 'workclass'),
        ('native-country', 'race'),
        ('earn', 'capital-remain'),
        ('workclass', 'earn'),
        ('sex', 'relationship'),
        ('occupation', 'earn'),
        ('hours-per-week', 'earn'),
        ('race', 'earn'),
        ('sex', 'earn')
    ])
    return model


def train(model, data, estimator):
    # 模型参数学习训练， 可以选择参数优化算法（最大似然估计、EM算法）
    model.fit(data, estimator=estimator, n_jobs=4)
    model.check_model()
    return model


def model():
    model = create_model()
    # 当数据集缺失率为0时，采用最大似然估计算法，其余情况使用EM算法
    # trained_model = train(model, train_data, MaximumLikelihoodEstimator)
    trained_model = train(model, train_data, ExpectationMaximization)
    return trained_model
