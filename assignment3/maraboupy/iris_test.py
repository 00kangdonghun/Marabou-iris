from maraboupy import Marabou
import numpy as np
import pandas as pd

# [수정] Marabou 옵션에서 verbosity=0으로 설정
options = Marabou.createOptions(verbosity=0)  # [수정] 로그 억제

# 1. 모델 및 데이터 로딩
model_path = "/home1/danny472/Marabou-1/assignment3/iris_simple.onnx"
csv_path = "/home1/danny472/Marabou-1/assignment3/iris.csv"

# ONNX MLP 모델 로드
net = Marabou.read_onnx(model_path)
inputVars = net.inputVars[0][0]        # 4 input nodes
outputVars = net.outputVars[0]         # [24, 25, 26] (3 output nodes)

# iris 데이터셋 읽기
df = pd.read_csv(csv_path, header=None).dropna()
X = df.iloc[:, :4].values.astype(np.float32)
y = df.iloc[:, 4].values
label_map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
y_num = np.array([label_map[label] for label in y])

# 2. 샘플별 예측 결과 및 정확도만 보기 좋게
correct = 0
for i, (x, y_true) in enumerate(zip(X, y_num)):
    input_sample = x.reshape(1, -1)
    out = net.evaluateWithMarabou([input_sample])[0]
    pred = np.argmax(out)
    if pred == y_true:
        correct += 1
print(f"\n[정확도] {correct}/{len(y_num)} = {correct/len(y_num)*100:.2f}%")

# --- 검증 공통 함수 ---
def check_sat_result(result, inputVars, msg_prefix):
    # Marabou 2023+ 버전은 결과가 ['sat'/'unsat'/'ERROR', dict, stat]로 옴
    status = result[0]
    vals = result[1]
    if status == "unsat":
        print(f"[{msg_prefix}] PASS: 조건을 항상 만족 (반례 없음)")
    elif status == "sat":
        print(f"[{msg_prefix}] FAIL: 반례 입력 존재!")
        # inputVars가 배열이라면, 입력값만 출력
        inp = [vals[v] for v in inputVars]
        print(f"    반례 입력값: {inp}")
    else:
        print(f"[{msg_prefix}] ERROR/Unknown: {status}")

# ------------------ Robustness 검증 ------------------
# 첫 샘플(x0) 주변 ±0.2 노이즈 범위에도 항상 setosa(0)로 분류되는가?
net = Marabou.read_onnx(model_path)
inputVars = net.inputVars[0][0]
outputVars = net.outputVars[0]
if len(outputVars) == 1 and hasattr(outputVars[0], '__len__'):
    outputVars = outputVars[0]

x0 = X[0]
noise = 0.2
for i in range(4):
    net.setLowerBound(inputVars[i], x0[i] - noise)
    net.setUpperBound(inputVars[i], x0[i] + noise)
net.addInequality([outputVars[0], outputVars[1]], [1, -1], 0)
net.addInequality([outputVars[0], outputVars[2]], [1, -1], 0)
result = net.solve()
check_sat_result(result, inputVars, "Robustness")

# ------------------ Safety 검증 ------------------
# 꽃받침 길이(0): 4.5~5.5, 꽃잎 길이(2): 1.0~1.5에서 항상 setosa(0)인가?
net = Marabou.read_onnx(model_path)
inputVars = net.inputVars[0][0]
outputVars = net.outputVars[0]
if len(outputVars) == 1 and hasattr(outputVars[0], '__len__'):
    outputVars = outputVars[0]

feature_bounds = [
    (4.5, 5.5),    # 0번
    (2.0, 4.4),    # 1번 (iris 전체 min/max)
    (1.0, 1.5),    # 2번
    (0.1, 2.5)     # 3번 (iris 전체 min/max)
]
for i, (lb, ub) in enumerate(feature_bounds):
    net.setLowerBound(inputVars[i], lb)
    net.setUpperBound(inputVars[i], ub)

net.addInequality([outputVars[0], outputVars[1]], [1, -1], 0)
net.addInequality([outputVars[0], outputVars[2]], [1, -1], 0)
result = net.solve(options=options)  
check_sat_result(result, inputVars, "Safety")


# ------------------ Fairness 검증 ------------------
# 첫 샘플 기준, 꽃받침 너비(1번 특성)만 3.0~3.7로 바꿔도 분류 변하지 않는가?
net = Marabou.read_onnx(model_path)
inputVars = net.inputVars[0][0]
outputVars = net.outputVars[0]
if len(outputVars) == 1 and hasattr(outputVars[0], '__len__'):
    outputVars = outputVars[0]
# 첫 샘플(x0)에서 1번만 바꾼다
net.setLowerBound(inputVars[0], x0[0])
net.setUpperBound(inputVars[0], x0[0])
net.setLowerBound(inputVars[1], 3.0)
net.setUpperBound(inputVars[1], 3.7)
net.setLowerBound(inputVars[2], x0[2])
net.setUpperBound(inputVars[2], x0[2])
net.setLowerBound(inputVars[3], x0[3])
net.setUpperBound(inputVars[3], x0[3])
net.addInequality([outputVars[0], outputVars[1]], [1, -1], 0)
net.addInequality([outputVars[0], outputVars[2]], [1, -1], 0)
result = net.solve()
check_sat_result(result, inputVars, "Fairness")
