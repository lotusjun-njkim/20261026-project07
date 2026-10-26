"""실습 04 기본필수: 이웃 3개의 투표와 평균을 비교합니다."""
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

X = [[1], [2], [4], [7], [9]]
class_y = [0, 0, 1, 1, 1]
number_y = [10, 20, 40, 70, 90]
# 3.2에 가까운 입력은 4, 2, 1입니다. 거리: 0.8, 1.2, 2.2
# 해설 B04a: 세 이웃의 정답 [1, 0, 0]에서 가장 많은 클래스 0을 선택합니다.
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, class_y)
print("분류:", classifier.predict([[3.2]]))
# 해설 B04b: (40 + 20 + 10) / 3 = 23.33입니다. 같은 이웃이라도 정답의 종류에 따라 계산이 달라집니다.
regressor = KNeighborsRegressor(n_neighbors=3)
regressor.fit(X, number_y)
print("회귀:", regressor.predict([[3.2]]).round(2))
