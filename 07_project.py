"""실습 07~08 기본필수: 같은 파일을 완성하고 다시 실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 파일 옆의 CSV를 읽고, 전날과 같은 요청들을 12 / 4 / 4로 나눕니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
develop, test = train_test_split(raw, test_size=0.2, random_state=42, stratify=raw["failed"])
train, valid = train_test_split(develop, test_size=0.25, random_state=42, stratify=develop["failed"])
# test는 보관만 합니다. 학습·예측에 넣지 않습니다.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score

# 제공: 전날의 두 입력을 그대로 사용합니다. elapsed_ms와 failed는 정답이므로 입력에서 뺍니다.
features = ["input_chars", "task_type"]
X_train = train[features]
X_valid = valid[features]
pre = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])
# 제공: 아래 두 반복은 같은 구조입니다. 첫 번째는 숫자, 두 번째는 클래스 정답을 사용합니다.
regression_rows = []
classification_rows = []
result = valid[["request_id", "elapsed_ms", "failed"]].copy()
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression())]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO B07a: 실행시간 elapsed_ms로 학습하고 X_valid를 예측하세요.
    # 힌트: model.fit(X_train, train["elapsed_ms"]) 다음 predict(X_valid)를 연결합니다.
    pass
    prediction = None
    regression_rows.append({"model": name, "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction)})
    result[name + "_ms"] = prediction

for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(max_iter=1000))]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO B07b: 실패 여부 failed로 학습하고 X_valid를 예측하세요.
    # 힌트: 회귀와 같은 흐름이며 정답 열만 failed로 바뀝니다.
    pass
    prediction = None
    classification_rows.append({"model": name, "F1": f1_score(valid["failed"], prediction, zero_division=0)})
    result[name + "_failed"] = prediction

# 제공: 두 종류의 지표를 섞지 않고 별도 표로 저장합니다. 저장 코드는 수정하지 않아도 됩니다.
regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)
regression.to_csv(HERE / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
result.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
print("분할:", len(train), len(valid), len(test))
print("회귀 비교 / 낮은 MAE가 좋음")
print(regression.round(3).to_string(index=False))
print("분류 비교 / 높은 F1이 좋음")
print(classification.round(3).to_string(index=False))
print("저장: regression_comparison.csv, classification_comparison.csv, validation_predictions.csv")
print("test는 학습·예측하지 않았습니다.")




"""실습 07~08 응용: 한 파일에서 회귀·분류 각각 세 후보를 비교합니다."""
from pathlib import Path
import pandas as pd

# [파일 경로 탐색] 현재 실행 중인 파이썬 스크립트 파일의 컴퓨터 내 실제 절대 경로와 부모 폴더 위치를 가져옵니다.
HERE = Path(__file__).resolve().parent

# [원본 데이터 로드] 'data/requests.csv' 파일을 읽어와 분석하고, 요청의 고유 ID('request_id')를 표의 행 이름(인덱스)으로 삼습니다.
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")

# [분할 설계도 로드] 데이터 세트를 어떻게 쪼갤지 전날 미리 적어둔 분할 계획서 파일('data/split_manifest.csv')을 읽어옵니다.
split = pd.read_csv(HERE / "data/split_manifest.csv")

# [학습용 ID 목록] 분할 계획서에서 'train'(학습용)에 배정된 요청 ID들을 찾아서 원래의 행 순서(position) 규칙대로 올바르게 정렬합니다.
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]

# [검증용 ID 목록] 분할 계획서에서 'valid'(검증용)에 배정된 요청 ID들을 찾아서 똑같이 원래 저장된 행 순서대로 똑바르게 정렬합니다.
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]

# [학습 데이터셋 생성] 정렬된 ID 순서에 맞춰서 원본 전체 데이터 중에서 실제 모델 훈련에 사용할 진짜 학습용 묶음(train)을 도출합니다.
train = raw.loc[train_ids]

# [검증 데이터셋 생성] 정렬된 ID 순서에 맞춰서 원본 전체 데이터 중에서 실제 모델 시험에 사용할 진짜 검증용 묶음(valid)을 도출합니다.
valid = raw.loc[valid_ids]

# [특징 변수 지정] 모델이 정답을 유추할 때 참고할 문제집의 힌트 정보(수치형 2개, 범주형 2개) 리스트를 정의합니다.
features = ["input_chars", "allowed_tools", "task_type", "priority"]

# [학습용 문제집] 훈련 데이터에서 모델 학습용 힌트로 지정한 4가지 특징 열(features)만 따로 분리하여 입력 데이터로 설정합니다.
X_train = train[features]

# [검증용 문제집] 검증 데이터에서 모델 시험용 힌트로 지정한 4가지 특징 열(features)만 따로 분리하여 시험 데이터로 설정합니다.
X_valid = valid[features]

# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 분석에 사용될 베이스라인 모델(Dummy), 선형 모델(Linear/Logistic), 거리 기반 모델(KNN), 그리고 각종 평가지표 함수들을 불러옵니다.
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
# [수치형 전처리 라인] 수치형 데이터용 가공 체인입니다. 빈 칸은 중앙값으로 메우고, 데이터 크기는 평균 0, 표준편차 1이 되도록 표준화합니다.
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

# [범주형 전처리 라인] 범주형 데이터용 가공 체인입니다. 빈 칸은 가장 자주 나온 단어로 채우고, 문자열은 컴퓨터가 이해하도록 0과 1의 벡터로 바꿉니다.
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

# [통합 전처리 공장] 앞서 구축한 숫자 가공 라인(처음 2개 열)과 글자 가공 라인(나머지 뒤의 2개 열)을 묶어 종합 전처리 변환기로 선언합니다.
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

# 제공 K: PPT 실습06의 동일 원본·분할·standard 탐색 결과(분류3, 회귀15)를 재사용합니다.
# 실험을 거친 뒤 각 모델의 최종 계산 성적표와 세부 예측 레코드를 보관할 빈 보관함(리스트)들을 선언합니다.
regression_rows = []
classification_rows = []
predictions = []

# ====================================================================================================
# [PART 1] 회귀(Regression) 분석 루프: 요청 처리에 걸리는 연속된 시간 수치('elapsed_ms') 예측 실험 시작
# ====================================================================================================
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression()),
                        ("KNN15", KNeighborsRegressor(n_neighbors=15))]:
    
    # [새로운 장비 결합] 전처리 공장의 오염을 막기 위해 깨끗하게 새로 복제(clone)한 전처리기 뒤에 현재 가동할 회귀 추정기(estimator)를 붙입니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    
    # [TODO A07r 반영] 학습용 문제집(X_train)과 실제 걸린 처리 시간 정답(train["elapsed_ms"])을 매칭하여 회귀 모델을 학습(fit)시킵니다.
    model.fit(X_train, train["elapsed_ms"])
    
    # [TODO A07r 반영] 학습이 완료된 회귀 모델 파이프라인에 검증용 문제집(X_valid)을 입력하여 예측 처리 시간 배열을 뽑아냅니다.
    prediction = model.predict(X_valid)
    
    # 검증 데이터의 진짜 걸린 시간과 모델의 예측 수치를 대조하여 실제 오차(MAE_ms, RMSE_ms) 지표를 계산하고 성적 보관함에 넣습니다.
    regression_rows.append({"model": name, "scale": "standard",
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
        "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)})
    
    # 추후 상세 검증 분석을 위해 검증 ID, 작업 유형(regression), 모델명, 실제값, 예측값을 하나의 판다스 표 객체로 묶어 누적 리스트에 기록합니다.
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "regression", "model": name,
        "actual": valid["elapsed_ms"].to_numpy(), "predicted": prediction}))

# ====================================================================================================
# [PART 2] 분류(Classification) 분석 루프: 요청의 최종 실패 여부('failed', 성공=0 / 실패=1) 이진 판정 실험 시작
# ====================================================================================================
for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(C=1.0, max_iter=1000)),
                        ("KNN3", KNeighborsClassifier(n_neighbors=3))]:
    
    # [새로운 장비 결합] 이번 분류 루프에서도 독립적인 실험을 보장하기 위해 전처리 공장을 원본 그대로 새로 복제하여 분류기 모델과 파이프라인으로 연결합니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    
    # [TODO A07c 반영] 학습용 문제집(X_train)과 실제 성공/실패 정답 코드(train["failed"])를 매칭하여 분류 모델을 학습(fit)시킵니다.
    model.fit(X_train, train["failed"])
    
    # [TODO A07c 반영] 학습이 끝난 분류 모델에 검증용 문제집(X_valid)을 풀어보게 하여 0(성공) 또는 1(실패)로 분류된 예측 클래스 배열을 도출합니다.
    prediction = model.predict(X_valid)
    
    # 오차 행렬(Confusion Matrix)을 통해 정탐과 오탐 건수를 계산합니다. 각각 TN, FP(과잉 오탐), FN(미탐 놓침), TP 건수를 반환합니다.
    tn, fp, fn, tp = confusion_matrix(valid["failed"], prediction, labels=[0, 1]).ravel()
    
    # 모델의 정밀도, 재현율, 종합 F1-Score 성적 및 틀린 개수(FP, FN)를 계산하여 분류 결과 성적 보관함에 딕셔너리로 저장합니다.
    classification_rows.append({"model": name, "scale": "standard",
        "F1": f1_score(valid["failed"], prediction, zero_division=0),
        "Precision": precision_score(valid["failed"], prediction, zero_division=0),
        "Recall": recall_score(valid["failed"], prediction, zero_division=0), "FP": fp, "FN": fn})
    
    # 개별 검증 건마다 모델이 진짜 정답(0 또는 1)을 뭐라고 예측했는지 상세 내역을 표 형태로 만들어 통합 예측 리스트에 기록합니다.
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "classification", "model": name,
        "actual": valid["failed"].to_numpy(), "predicted": prediction}))

# 보관함에 하나씩 모아두었던 실험 데이터 덩어리들을 분석과 정렬이 용이한 최종 판다스 데이터프레임(표) 구조로 변환합니다.
regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)

# 성적표 파일들을 내보낼 저장 위치인 'outputs' 폴더가 현재 부모 경로에 존재하는지 체크하고, 없으면 자동으로 생성합니다.
out = HERE / "outputs"
out.mkdir(exist_ok=True)

# 엑셀 프로그램 등에서 한국어가 깨지지 않고 온전하게 잘 열리도록 'utf-8-sig' 인코딩 포맷을 명시하여 비교 표 파일 2개를 디스크에 기록합니다.
regression.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(out / "classification_comparison.csv", index=False, encoding="utf-8-sig")

# 모든 모델(회귀 3종, 분류 3종)의 모든 행별 세부 예측 결과 데이터를 수직으로 길게 이어 붙여 하나의 대형 CSV 리포트 파일로 내보냅니다.
pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")

# 모든 가동이 끝난 후, 터미널 모니터 창에 최종 취합된 회귀 비교 결과 표를 소수점 3자리까지 예쁘게 다듬어서 출력합니다.
print("회귀 / valid 20건")
print(regression.round(3).to_string(index=False))

# 터미널 모니터 창에 최종 취합된 분류 성능 지표 결과 표를 똑같이 소수점 3자리까지 정렬하여 보기 좋게 출력합니다.
print("분류 / valid 20건")
print(classification.round(3).to_string(index=False))

# 파일이 유실 없이 정상 저장되었음을 알리고, 데이터 과학 규칙에 맞춰 테스트 세트(Test set)는 철저히 보안을 유지하여 건드리지 않았음을 선언합니다.
print("저장: outputs의 비교표 2개와 validation_predictions.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
