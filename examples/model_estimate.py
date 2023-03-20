from pgmpy.estimators import HillClimbSearch, K2Score
from parse_dataset import preprocess

train_data = preprocess("../data/adult/adult.data", 0)

# 模型结构学习，方便更精准的创建模型
scoring_method = K2Score(data=train_data)
est = HillClimbSearch(data=train_data)
model = est.estimate(scoring_method=scoring_method, max_iter=int(1e4))

print(model.edges())
