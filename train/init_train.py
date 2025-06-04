# init_train.py
from sentence_transformers import SentenceTransformer, InputExample, losses
import torch
import os

# 기본 사전학습 모델로 시작
base_model = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(base_model)

# 학습 예시 데이터 구성
train_examples = [
    InputExample(texts=["tomato sauce", "pizza sauce"], label=1),
    InputExample(texts=["banana puree", "fruit paste"], label=1),
    InputExample(texts=["tomato sauce", "banana puree"], label=0),
]

# 손실 함수 및 트레이너 구성
train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=2)
train_loss = losses.CosineSimilarityLoss(model)

# 학습 시작
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10)

# 학습한 모델 저장 (폴더 수정: models/trained_model)
save_path = os.path.abspath("models/trained_model")
model.save(save_path)
