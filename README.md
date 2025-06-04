# Marabou Neural Network Verification – Iris MLP Example

본 프로젝트는 [Marabou](https://github.com/NeuralNetworkVerification/Marabou)를 활용해 PyTorch로 학습한 Iris 분류 MLP 모델의 **정확도**, **강인성(Robustness)**, **안전성(Safety)**, **공정성(Fairness)**을 자동 검증하는 실습 예제입니다.

---

## 구성 파일

- `iris_simple.py`  : MLP 모델 학습 및 ONNX 변환
- `iris_test.py`   : Marabou 기반 검증 코드 (정확도, Robustness, Safety, Fairness)
- `iris_simple.onnx` : ONNX 변환된 모델 (실행 시 자동 생성)
- `iris.csv`     : Iris 데이터셋 (scikit-learn 사용)
- `requirements.txt` : 실행 환경 패키지 목록

---

##  ⚙️ Setup

```bash
pip install -r requirements.txt
```
### 1. Clone the repository

```bash
git clone https://github.com/your-username/adversarial-attacks.git
cd assignment3
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure
```
.
├── assignment3/
│   └── maraboupy
│       └── iris_test.py           
│   └── iris_simple.onxx
│   └── iris_simple.py
│   └── iris.csv           
├── bulid/
    ├── cmake-3.31.7          
    └── cmake-3.31.7.tar.gz       
```

## 🚀 Usage

### 1. cd assignment3
```bash
cd assignmnet3
```

### 2. convert ONNX
```bash
python iris_simple.py
```

### 3. test Marabou_iris
```bash
python maraboupy/marabou_iris_test.py
```

## 📊 Results 

```
[정확도] 146/150 = 97.33%
unsat
[Robustness] PASS: 조건을 항상 만족 (반례 없음)
sat
input 0 = 4.5
input 1 = 2.2305695652271846
input 2 = 1.0
input 3 = 2.5
output 0 = 0.5952426074340558
output 1 = 1.1542043262442125
output 2 = 0.5952426074340559
[Safety] FAIL: 반례 입력 존재!
    반례 입력값: [4.5, 2.2305695652271846, 1.0, 2.5]
unsat
[Fairness] PASS: 조건을 항상 만족 (반례 없음)
Engine::processInputQuery: Input query (before preprocessing): 15 equations, 27 variables
```
