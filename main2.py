import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import datetime

# 파일 경로 설정
train_path = "/Users/nextweb/Downloads/open/train.csv"
test_path = "/Users/nextweb/Downloads/open/test.csv"
submission_path = "/Users/nextweb/Downloads/open/sample_submission.csv"

# 파일 존재 확인
for fpath in [train_path, test_path, submission_path]:
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"{fpath} 파일이 존재하지 않습니다.")

# CSV 파일 불러오기
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)

# 컬럼 이름 확인
if 'full_text' not in train_df.columns or 'generated' not in train_df.columns:
    raise KeyError("'train.csv'에는 'full_text'와 'generated' 컬럼이 있어야 합니다.")
if 'paragraph_text' not in test_df.columns:
    raise KeyError("'test.csv'에는 'paragraph_text' 컬럼이 있어야 합니다.")

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2
)
X_train = vectorizer.fit_transform(train_df['full_text'])
y_train = train_df['generated']

# XGBoost 모델 학습 (random_state 설정하지 않음 → 매번 결과 달라짐)
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 테스트 데이터 예측
X_test = vectorizer.transform(test_df['paragraph_text'])
test_preds = model.predict_proba(X_test)[:, 1]  # 확률값 추출

# 예측 결과를 확률 그대로 제출
submission['generated'] = test_preds

# 파일 이름에 현재 날짜/시간 추가
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"/Users/nextweb/Downloads/open/submission_{now}.csv"
submission.to_csv(output_path, index=False, encoding='utf-8')

print(f"✅ 제출용 파일 저장 완료: {output_path}")