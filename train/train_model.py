from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import pandas as pd
import os

# 1. 모델 불러오기 (기반은 all-mpnet-base-v2)
base_model = os.path.abspath("trained_model")
model_save_path = "fine_tuned_model"

model = SentenceTransformer(base_model)

# 2. 훈련 데이터 로드
df = pd.read_csv("training_data.csv")
train_samples = []

for _, row in df.iterrows():
    train_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=1.0))
1
# 3. 데이터 로더 및 loss 구성
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# 4. 훈련
num_epochs = 1  # 필요시 2~3회 반복 가능
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
