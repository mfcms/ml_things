import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
samples = pd.read_csv('yandex_test/samples.csv', sep='\t')
data = pd.read_csv('yandex_test/data.csv', sep='\t')

# Функция для извлечения сущностей и контекста
def extract_entities_and_context(text):
    # Извлечение тегов сущностей
    entities = re.findall(r'<e\d+>(.*?)</e\d+>', text)
    if len(entities) != 2:
        return None, None, None
    
    # Извлечение самих тегов (например, <e1>, <e2>)
    tags = re.findall(r'<(e\d+)>', text)
    if len(tags) != 2:
        return None, None, None
    
    # Извлечение текста между сущностями
    parts = re.split(r'<e\d+>.*?</e\d+>', text)
    context_between = parts[1] if len(parts) > 1 else ""
    
    # Извлечение текста до первой сущности и после второй
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
        
        # Базовые признаки
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
        }
        
        # Дополнительные лингвистические признаки
        context_text = ' '.join(context_parts)
        feature_dict['has_preposition'] = int(any(p in context_text.lower() for p in ['в', 'на', 'к', 'с', 'у', 'из', 'от', 'по', 'за', 'до']))
        feature_dict['has_conjunction'] = int(any(c in context_text.lower() for c in ['и', 'а', 'но', 'или', 'да', 'же', 'тоже', 'также']))
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)

# Обработка обучающих данных
train_features = create_features(samples)
train_labels = samples['label'].copy()

# Создание TF-IDF векторов для текстовых признаков
vectorizers = {}
text_columns = ['entity1', 'entity2', 'context_before', 'context_between', 'context_after']

for col in text_columns:
    vectorizers[col] = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizers[col].fit_transform(train_features[col].fillna('').astype(str))
    # Преобразуем в DataFrame и добавляем к основным признакам
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                           columns=[f"{col}_{i}" for i in range(tfidf_matrix.shape[1])],
                           index=train_features.index)
    train_features = pd.concat([train_features, tfidf_df], axis=1)
    train_features.drop(col, axis=1, inplace=True)

# Подготовка данных для обучения
X_train = train_features.drop(['tag1', 'tag2'], axis=1, errors='ignore')  # Убираем теги из признаков
y_train = train_labels

# Обучение модели
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Обработка тестовых данных
test_features = create_features(data)

# Применение TF-IDF векторизаторов к тестовым данным
for col in text_columns:
    if col in test_features.columns:
        tfidf_matrix = vectorizers[col].transform(test_features[col].fillna('').astype(str))
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                               columns=[f"{col}_{i}" for i in range(tfidf_matrix.shape[1])],
                               index=test_features.index)
        test_features = pd.concat([test_features, tfidf_df], axis=1)
        test_features.drop(col, axis=1, inplace=True)

# Подготовка тестовых данных
X_test = test_features.drop(['tag1', 'tag2'], axis=1, errors='ignore')

# Предсказание
predictions = model.predict(X_test)

# Создание выходного файла
output_df = pd.DataFrame({'label': predictions})
output_df.to_csv('yandex_test/predictions.csv', index=False)

print("Модель обучена и предсказания сохранены в predictions.csv")

prov = pd.read_csv('yandex_test/predictions.csv')
print("Столбцы predictions.csv:", list(prov.columns))
print("Первые 5 строк:")
print(prov.head())