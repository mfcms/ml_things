import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# --- Функция обработки ---
def preprocess_dataframe(df, is_train=True, encoders=None):
    df = df.copy()
    numeric_cols = []
    cat_cols = []
    new_encoders = {} if encoders is None else encoders

    for col in df.columns:
        if col == 'target' and not is_train:
            continue

        series_str = df[col].astype(str).str.replace(',', '.', regex=False)
        numeric_series = pd.to_numeric(series_str, errors='coerce')

        if numeric_series.notna().mean() >= 0.9:
            numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
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

# --- Загрузка и обработка ---
train = pd.read_csv('yandex_test/train.csv')
test = pd.read_csv('yandex_test/test.csv')

# Сохраняем оригинальный порядок customer (если есть ID — берем его, иначе — индексы)
if 'id' in test.columns:
    customer_ids = test['id'].values
elif 'customer' in test.columns:
    customer_ids = test['customer'].values
else:
    customer_ids = range(len(test))

train_processed, encoders = preprocess_dataframe(train, is_train=True)
test_processed, _ = preprocess_dataframe(test, is_train=False, encoders=encoders)

X_train = train_processed.drop(columns=['target'])
y_train = train_processed['target'].astype(int)
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
    'customer': customer_ids,  # ← ПРАВИЛЬНО: соответствует порядку в test.csv
    'target': y_test_pred.astype(int)
})
answers.to_csv('yandex_test/answers.csv', index=False)

print("✅ Файл answers.csv сохранён в правильном формате.")

test = pd.read_csv('yandex_test/test.csv')
print("Столбцы test.csv:", list(test.columns))
print("Первые 5 строк:")
print(test.head())

prov = pd.read_csv('yandex_test/answers.csv')
print("Столбцы test.csv:", list(prov.columns))
print("Первые 5 строк:")
print(prov.head())