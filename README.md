# dacon-llm-challenge2
dacon-llm-challenge2
# 🧠 AI-Generated Text Classifier using TF-IDF and XGBoost

이 프로젝트는 DACON의 생성 AI 텍스트 판별 챌린지를 위한 베이스라인 모델로, 기계 학습 알고리즘(XGBoost)과 텍스트 피처링(TF-IDF)을 기반으로 AI 생성 텍스트를 분류하는 시스템입니다.

## 📁 프로젝트 구조

```
.
├── main.py                # 실행 코드 (모델 학습 + 예측 + 제출 파일 생성)
├── train.csv              # 학습 데이터
├── test.csv               # 테스트 데이터
├── sample_submission.csv # 제출 양식
└── README.md              # 프로젝트 설명 파일
```

## 📊 데이터 설명

### 🔹 train.csv
| 컬럼명       | 설명                          |
|--------------|-------------------------------|
| `full_text`  | 전체 문단 (입력 텍스트)       |
| `generated`  | 생성 여부 (1: AI 생성, 0: 인간 작성) |

### 🔹 test.csv
| 컬럼명         | 설명                         |
|----------------|------------------------------|
| `paragraph_text` | 전체 문단 (예측 대상)        |

### 🔹 sample_submission.csv
| 컬럼명       | 설명                          |
|--------------|-------------------------------|
| `id`         | 샘플 ID                        |
| `generated`  | 예측값 (0~1 사이 확률 값)     |

## ⚙️ 실행 방법

1. 필요한 라이브러리 설치:
```bash
pip install pandas scikit-learn xgboost
```

2. `main.py` 실행:
```bash
python main.py
```

3. 실행 후 아래 경로에 결과 파일이 생성됩니다:
```
/Users/nextweb/Downloads/open/submission_YYYYMMDD_HHMMSS.csv
```

## 🧪 사용된 기술

- **언어 모델링**: TF-IDF 기반 벡터화
- **분류기**: XGBoost Classifier
- **피처 엔지니어링**: n-gram(1~2), 불용어 제거, min_df, max_features 등 적용

## 🧠 모델 하이퍼파라미터

```python
TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2
)

XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)
```

## 📈 성능 평가

- Validation AUC 기준 최고 약 **0.63**
- 기본 베이스라인 대비 향상 가능성 있음
- 추후 KoBERT / KoELECTRA 등으로 모델 교체 시 추가 개선 기대

## 📌 참고 사항

- 실행 시 `train.csv`, `test.csv`, `sample_submission.csv`가 지정 경로에 있어야 합니다.
- 제출 파일에는 `generated` 컬럼에 확률값(0~1)을 그대로 저장합니다.

## 📄 라이선스

MIT License

## 🙋 기여

궁금한 점이 있거나 함께 개선하고 싶다면 [Issues](https://github.com/your-repo/issues) 혹은 PR 환영합니다!
