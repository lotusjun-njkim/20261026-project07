"""실습 01 기본필수: 학습과 예측 두 문장을 연결합니다."""
from sklearn.linear_model import LinearRegression

# X는 한 행이 한 사람이고, 각 행의 숫자는 공부시간입니다. 그래서 [[1], [2], ...]입니다.
X = [[1], [2], [3], [4], [5]]
y = [52, 61, 69, 80, 89]  # 같은 순서의 실제 점수
model = LinearRegression()  # 선을 맞추는 모델을 준비합니다. 아직 학습 전입니다.
# 해설 B01: fit은 계수·절편을 학습합니다. predict([[6]])는 새 입력 한 행의 값을 배열로 반환합니다.
model.fit(X, y)
prediction = model.predict([[6]])
print("기울기:", round(model.coef_[0], 1))
print("절편:", round(model.intercept_, 1))
print("6시간 예측:", round(prediction[0], 1))
# 주의: 6시간은 학습 범위 1~5시간 밖입니다. 수식 확인용이며 실제 점수를 보장하지 않습니다.
