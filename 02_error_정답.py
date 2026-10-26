"""실습 02 기본필수: 미리 제공된 예측으로 MAE와 RMSE를 계산합니다."""
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

actual = [100, 200, 300]     # 실제 소요시간, 단위 ms
prediction = [120, 180, 360] # 모델이 예측했다고 가정한 값, 단위 ms
# 해설 B02: MAE는 평균 절대오차이고 RMSE는 제곱오차 평균의 제곱근입니다. 둘 다 정답과 같은 ms 단위입니다.
mae = mean_absolute_error(actual, prediction)
rmse = root_mean_squared_error(actual, prediction)
print("MAE(ms):", round(mae, 2))
print("RMSE(ms):", round(rmse, 2))
# MAE는 절댓값 20·20·60의 평균입니다. RMSE는 제곱 400·400·3600의 평균에 루트를 취합니다.
