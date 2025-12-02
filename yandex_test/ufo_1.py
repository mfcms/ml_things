import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных (предполагаем, что разделитель — табуляция)
try:
    samples = pd.read_csv('yandex_test/samples.csv', sep='\t')
    data = pd.read_csv('yandex_test/data.csv', sep='\t')
except FileNotFoundError:
    # Альтернатива: если файлы в текущей директории
    samples = pd.read_csv('samples.csv', sep='\t')
    data = pd.read_csv('data.csv', sep='\t')

# Функция для извлечения сущностей и контекста
def extract_entities_and_context(text):
    entities = re.findall(r'<e\d+>(.*?)</e\d+>', text)
    if len(entities) != 2:
        return None, None, None
    
    tags = re.findall(r'<(e\d+)>', text)
    if len(tags) != 2:
        return None, None, None
    
    parts = re.split(r'<e\d+>.*?</e\d+>', text)
    context_between = parts[1] if len(parts) > 1 else ""
    before_first = parts[0] if len(parts) > 0 else ""
    after_last = parts[2] if len(parts) > 2 else ""
    
    return entities, tags, (before_first, context_between, after_last)

# Создание признаков
def create_features(df):
    features = []
    for idx, row in df.iterrows():
        text = row['text']
        entities, tags, context_parts = extract_entities_and_context(text)
        
        if entities is None or tags is None:
            features.append({
                'entity1': '',
                'entity2': '',
                'tag1': '',
                'tag2': '',
                'context_before': '',
                'context_between': '',
                'context_after': '',
                'sentence_length': 0,
                'num_words': 0,
                'has_preposition': 0,
                'has_conjunction': 0
            })
            continue
        
        context_text = ' '.join(context_parts)
        feature_dict = {
            'entity1': entities[0],
            'entity2': entities[1],
            'tag1': tags[0],
            'tag2': tags[1],
            'context_before': context_parts[0],
            'context_between': context_parts[1],
            'context_after': context_parts[2],
            'sentence_length': len(text),
            'num_words': len(text.split()),
            'has_preposition': int(any(p in context_text.lower() for p in ['в', 'на', 'к', 'с', 'у', 'из', 'от', 'по', 'за', 'до'])),
            'has_conjunction': int(any(c in context_text.lower() for c in ['и', 'а', 'но', 'или', 'да', 'же', 'тоже', 'также'])),
        }
        features.append(feature_dict)
    
    return pd.DataFrame(features)

# Обработка обучающих данных
train_features = create_features(samples)
train_labels = samples['label'].copy()

# Разделение на train и validation (80/20)
X_train_feat, X_val_feat, y_train, y_val = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

# TF-IDF векторизация
vectorizers = {}
text_columns = ['entity1', 'entity2', 'context_before', 'context_between', 'context_after']

# Обучение векторизаторов на train, применение к train и val
X_train_processed = X_train_feat.copy()
X_val_processed = X_val_feat.copy()

for col in text_columns:
    vectorizer = TfidfVectorizer(max_features=500, stop_words=None, lowercase=True)
    # Обучаем на train
    tfidf_train = vectorizer.fit_transform(X_train_feat[col].fillna('').astype(str))
    tfidf_val = vectorizer.transform(X_val_feat[col].fillna('').astype(str))
    
    # Преобразуем в DataFrame
    tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), 
                                  columns=[f"{col}_tfidf_{i}" for i in range(tfidf_train.shape[1])],
                                  index=X_train_feat.index)
    tfidf_val_df = pd.DataFrame(tfidf_val.toarray(),
                                columns=[f"{col}_tfidf_{i}" for i in range(tfidf_val.shape[1])],
                                index=X_val_feat.index)
    
    # Добавляем к данным
    X_train_processed = pd.concat([X_train_processed, tfidf_train_df], axis=1)
    X_val_processed = pd.concat([X_val_processed, tfidf_val_df], axis=1)
    
    # Удаляем исходный текстовый столбец
    X_train_processed.drop(col, axis=1, inplace=True, errors='ignore')
    X_val_processed.drop(col, axis=1, inplace=True, errors='ignore')
    
    vectorizers[col] = vectorizer

# Убираем теги (они не несут полезной информации для обобщения)
X_train_final = X_train_processed.drop(['tag1', 'tag2'], axis=1, errors='ignore')
X_val_final = X_val_processed.drop(['tag1', 'tag2'], axis=1, errors='ignore')

# Обучение модели
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_final, y_train)

# Предсказание на валидации
val_preds = model.predict(X_val_final)

# Расчёт и вывод Macro F1-score
macro_f1 = f1_score(y_val, val_preds, average='macro')
print(f"Macro F1-score на валидационной выборке: {macro_f1:.4f}")

# Также можно вывести подробный отчёт
print("\nПодробный отчёт по классам:")
print(classification_report(y_val, val_preds))

# Теперь обучаем финальную модель на ВСЕХ обучающих данных для предсказания на test
# --- Финальная модель на полных данных ---
X_full_processed = train_features.copy()
for col in text_columns:
    tfidf_full = vectorizers[col].transform(train_features[col].fillna('').astype(str))
    tfidf_full_df = pd.DataFrame(tfidf_full.toarray(),
                                 columns=[f"{col}_tfidf_{i}" for i in range(tfidf_full.shape[1])],
                                 index=train_features.index)
    X_full_processed = pd.concat([X_full_processed, tfidf_full_df], axis=1)
    X_full_processed.drop(col, axis=1, inplace=True, errors='ignore')

X_full_final = X_full_processed.drop(['tag1', 'tag2'], axis=1, errors='ignore')
final_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
final_model.fit(X_full_final, train_labels)

# Обработка тестовых данных
test_features = create_features(data)
X_test_processed = test_features.copy()

for col in text_columns:
    if col in test_features.columns:
        tfidf_test = vectorizers[col].transform(test_features[col].fillna('').astype(str))
        tfidf_test_df = pd.DataFrame(tfidf_test.toarray(),
                                     columns=[f"{col}_tfidf_{i}" for i in range(tfidf_test.shape[1])],
                                     index=test_features.index)
        X_test_processed = pd.concat([X_test_processed, tfidf_test_df], axis=1)
        X_test_processed.drop(col, axis=1, inplace=True, errors='ignore')

X_test_final = X_test_processed.drop(['tag1', 'tag2'], axis=1, errors='ignore')

# Предсказание на тесте
test_preds = final_model.predict(X_test_final)

# Сохранение результата
output_df = pd.DataFrame({'label': test_preds})
output_df.to_csv('predictions.csv', index=False)

print("\nПредсказания на тестовом наборе сохранены в 'predictions.csv'")