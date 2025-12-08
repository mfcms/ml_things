from ast import Lambda
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow_hub as hub  # для предобученных эмбеддингов
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('ml_things/fake_news/FakeNewsNet.csv')

# ДАТАСЕТ ОКАЗАЛСЯ КРИВЫМ

df['source_domain'] = df['source_domain'].fillna('unknown').astype(str)
df['news_url'] = df['news_url'].fillna('').astype(str)
df['title'] = df['title'].fillna('').astype(str)
df = df.drop_duplicates().reset_index(drop=True)

print("После очистки:")
print(df['source_domain'].isna().sum(), "пропусков в source domain")
print(df['source_domain'].value_counts().head())

df = df.drop_duplicates().dropna(subset=['title', 'real'])

df['real'] = df['real'].astype(int)

print(f"Всего: {len(df)} статей | Фейков: {len(df[df['real']==0])} ({df['real'].mean():.1%} реальных)")

import re
def extract_domain(url_or_domain):
    if pd.isna(url_or_domain) or url_or_domain is None:
        return "unknown"  # или "missing", "" — как вам удобнее
    
    url_or_domain = str(url_or_domain).strip()
    
    if not url_or_domain:
        return "unknown"
    
    if url_or_domain.startswith(('http://', 'https://')):
        match = re.search(r'https?://(?:www\.)?([^/:\s]+)', url_or_domain)
        domain = match.group(1) if match else url_or_domain
    else:
        domain = url_or_domain
    
    domain = re.split(r'[:/?#]', domain)[0]
    
    parts = domain.split('.')
    if len(parts) >= 2:
        return domain.lower()
    else:
        return domain.lower()

df['clean_domain'] = df['source_domain'].apply(extract_domain)

le_domain = LabelEncoder()
df['domain_encoded'] = le_domain.fit_transform(df['clean_domain'])

n_domains = len(le_domain.classes_)
print(f"Уникальных доменов: {n_domains}")

scaler_tweet = StandardScaler()
df['tweet_scaled'] = scaler_tweet.fit_transform(df[['tweet_num']])

X_text = df['title'].values 
X_domain = df['domain_encoded'].values
X_tweet = df['tweet_scaled'].values
y = df['real'].values

# Разбиение
(
    X_text_train, X_text_val,
    X_domain_train, X_domain_val,
    X_tweet_train, X_tweet_val,
    y_train, y_val
) = train_test_split(
    X_text, X_domain, X_tweet, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def USEEmbeddingLayer():
    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    def call(inputs):
        return use(inputs)
    return Lambda(call, output_shape=(512,), name="USE")

text_input = Input(shape=(), dtype=tf.string, name="title")
text_emb = USEEmbeddingLayer()(text_input)  # ← так работает во всех T
text_dense = Dense(128, activation='relu')(text_emb)
text_out = Dropout(0.3)(text_dense)

domain_input = Input(shape=(1,), name="domain")
# Embedding: каждому домену — вектор размера 8
domain_emb = Embedding(input_dim=n_domains, output_dim=8)(domain_input)
domain_emb = Flatten()(domain_emb)
domain_out = Dense(16, activation='relu')(domain_emb)

tweet_input = Input(shape=(1,), name="tweet_num")
tweet_out = Dense(8, activation='relu')(tweet_input)

combined = concatenate([text_out, domain_out, tweet_out])
x = Dense(64, activation='relu')(combined)
x = Dropout(0.4)(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid', name="real")(x)  # бинарная классификация

model = Model(
    inputs=[text_input, domain_input, tweet_input],
    outputs=output
)

model.compile(
    optimizer=Adam(learning_rate=2e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_auc',
    patience=5,
    restore_best_weights=True,
    mode='max'
)

history = model.fit(
    {
        "title": X_text_train,
        "domain": X_domain_train,
        "tweet_num": X_tweet_train
    },
    y_train,
    validation_data=({
        "title": X_text_val,
        "domain": X_domain_val,
        "tweet_num": X_tweet_val
    }, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Предсказания на валидации
y_pred_proba = model.predict({
    "title": X_text_val,
    "domain": X_domain_val,
    "tweet_num": X_tweet_val
}).flatten()

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

auc_val = roc_auc_score(y_val, y_pred_proba)
print(f"Validation ROC AUC: {auc_val:.4f}")

# Если нужен бинарный вывод (0/1)
y_pred = (y_pred_proba > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['Fake', 'Real']))