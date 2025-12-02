import pandas as pd
import re
from sklearn import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

train = pd.read_csv('titanic_surv/train.csv')
test = pd.read_csv('titanic_surv/test.csv')

def preprocess_dataframe(df, is_train=True, encoders=None):
    df = df.copy()
    numeric_cols = []
    cat_cols = []
    new_encoders = {} if encoders is None else encoders # wtf

    for col in df.columns:
        series_str = df[col].astype(str).str.replace(',', '.', regex=False)
        numeric_series = pd.to_numeric(series_str, errors='coerce')

        if numeric_series.notna().mean() >= 0.9:
            median_val = numeric_series.median()
            numeric_series.fillna(median_val, inplace=True)
            df[col] = numeric_series
            numeric_cols.append(col)
        else:
            df[col] = df[col].astype(str)
            cat_cols.append(col)

        if encoders is None:
            if cat_cols:
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                df[cat_cols] = encoder.fit_transform(df[cat_cols])
                new_encoders['encoder'] = encoder
                new_encoders['cat_cols'] = cat_cols
        else:
            cat_cols = encoders.get('cat_cols', [])
            if cat_cols:
                for col in cat_cols:
                    df[col] = df[col].astype(str)
                df[cat_cols] = encoders['encoder'].transform(df[cat_cols])

    return df, new_encoders

if 'PassengerId' in test.columns:
    customer_ids = test['PassengerId'].values
elif 'Name' in test.columns:
    customer_ids = test['Name'].values
else:
    customer_ids = range(len(test))

train_processed, encoders = preprocess_dataframe(train, is_train=True)
test_processed, _ = preprocess_dataframe(test, is_train=False, encoders=encoders)

X_train = train_processed.drop(columns=['Survived'])
y_train = train_processed['Survived'].astype(int)
X_test = test_processed.copy()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Target distribution:\n", y_train.value_counts(normalize=True))

# Разбиение
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Балансировка
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Модель
model = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50
)

model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

# Оценка
y_val_proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_proba)
print(f"Validation ROC AUC: {auc:.4f}")

# Предсказание
y_test_pred = model.predict(X_test)

# Сохранение в правильном формате
answers = pd.DataFrame({
    'PassengerId': customer_ids,  # ← ПРАВИЛЬНО: соответствует порядку в test.csv
    'survived': y_test_pred.astype(int)
})
answers.to_csv('titanic_surv/answers.csv', index=False)