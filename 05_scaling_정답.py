"""실습 05 기본필수: train에서 정한 기준을 새 입력에도 적용합니다."""
from sklearn.preprocessing import StandardScaler

train = [[100], [200], [300]]
valid = [[400]]
scaler = StandardScaler()
scaler.fit(train)  # 제공: 평균과 크기 기준은 train에서만 구합니다.
# 해설 B05: transform은 train에서 학습한 평균과 표준편차를 재사용합니다. valid를 변환해도 평균 200은 바뀌지 않습니다.
converted = scaler.transform(valid)
print("train 기준 평균:", scaler.mean_)
print("train 변환:", scaler.transform(train).round(3))
print("valid 변환:", converted.round(3))
print("변환 후 기준 평균:", scaler.mean_)
# valid에 다시 fit하지 않습니다. StandardScaler는 자료를 정규분포로 만들지 않습니다.
