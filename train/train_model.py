"""
train_model.py
--------------

사전학습된 문장 임베딩 모델을 사용자 피드백 데이터로 파인튜닝(fine-tuning)하고, 업데이트된 모델 저장

1. 기존 사전학습 모델 로드 (models/trained_model)
2. 사용자 피드백 기반 훈련 데이터 로드 (data/training_data.csv)
3. 데이터로더 및 손실 함수 설정
4. 모델 파인튜닝 학습
5. 파인튜닝된 모델 저장 (models/trained_model)

"""

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os

# 1. 모델 불러오기 (기존 사전학습 모델)
base_model = "models/trained_model"
model_save_path = "models/trained_model"

model = SentenceTransformer(base_model)

# 2. 훈련 데이터 로드
df = pd.read_csv("data/training_data.csv")
train_samples = []

for _, row in df.iterrows():
    train_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=1.0))

# 3. 데이터로더 및 손실 함수 구성
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# 4. 모델 학습 (fine-tuning)
num_epochs = 1  # 필요 시 2~3회 반복 가능
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True
)

# 5. 모델 저장
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
model.save(model_save_path)
print(f"파인튜닝된 모델 저장 완료: {model_save_path}")
