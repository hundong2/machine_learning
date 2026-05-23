import seaborn as sns
import matplotlib.pyplot as plt

# 수치형 변수 분포 확인
numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()[:-1]

for feature in numerical_features:
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    sns.histplot(train[feature], kde=True)
    plt.title('Distribution of ' + feature)
    plt.subplot(1, 2, 2)
    sns.boxplot(y=train[feature])
    plt.title('Boxplot of ' + feature)
    plt.show()