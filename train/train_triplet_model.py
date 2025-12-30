"""
train_triplet_model.py
----------------------
단일 triplet 데이터셋 기반 SentenceTransformer 모델 파인튜닝 스크립트.
KR/EN 분리 없이 하나의 모델만 지속적으로 업데이트하는 구조.
"""

import os
import sys
import pandas as pd
import torch
import argparse
import datetime
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def count_new_feedback(path=None):
    if path is None:
        path = str(config.USER_FEEDBACK_CSV)
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path)
    if "status" not in df.columns:
        return 0
    return len(df[df["status"] == "NEW"])


def train_triplet_model(
    triplet_path=None,
    trained_model_path=None,
    epochs=None,
    batch_size=None,
    learning_rate=None
):
    """
    Triplet 데이터를 사용하여 SentenceTransformer 모델을 학습합니다.
    
    Args:
        triplet_path: Triplet 학습 데이터 경로 (기본값: config.TRIPLET_TRAINING_DATA_CSV)
        trained_model_path: 모델 저장 경로 (기본값: config.TRAINED_MODEL_PATH)
        epochs: 학습 epoch 수 (기본값: config.EPOCH_CONFIG에 따라 자동 결정)
        batch_size: 배치 크기 (기본값: config.BATCH_SIZE)
        learning_rate: 학습률 (기본값: config.LEARNING_RATE)
    """
    if triplet_path is None:
        triplet_path = str(config.TRIPLET_TRAINING_DATA_CSV)
    if trained_model_path is None:
        trained_model_path = str(config.TRAINED_MODEL_PATH)
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    
    # ============================================
    # Device 설정
    # ============================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스 사용: {device}")
    
    if device == "cuda":
        print(f"GPU 정보: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # GPU가 있고 배치 사이즈가 기본값이면 GPU용 배치 사이즈로 자동 조정
        if batch_size == config.BATCH_SIZE:
            batch_size = config.BATCH_SIZE_GPU
            print(f"[INFO] GPU 감지: 배치 사이즈를 {config.BATCH_SIZE}에서 {batch_size}로 자동 증가 (더 빠른 학습)")

    # ============================================
    # Step 1: 모델 로드 (Continual Learning)
    # ============================================
    # 재학습 구조: 기존 모델이 있으면 계속 학습 (지식 누적)
    # 없으면 기본 모델로 시작
    if os.path.exists(trained_model_path) and os.listdir(trained_model_path):
        print(f"[MODEL] 기존 학습된 모델 로드 → {trained_model_path}")
        print(f"[MODEL] Continual Learning: 기존 모델에 추가 학습 진행")
        model = SentenceTransformer(trained_model_path, device=device)
    else:
        print(f"[MODEL] Fine-tuning 시작용 초기 모델 로드: {config.DEFAULT_MODEL_NAME}")
        print(f"[MODEL] 첫 학습: 기본 모델로 시작")
        model = SentenceTransformer(
            config.DEFAULT_MODEL_NAME,
            device=device
        )

    # ============================================
    # Step 2: Triplet 데이터 로드
    # ============================================
    print(f"\n[DATA] Triplet 학습 데이터 로드 중: {triplet_path}")

    if not os.path.exists(triplet_path):
        raise FileNotFoundError(f"[ERROR] Triplet 데이터 파일 없음: {triplet_path}")

    df = pd.read_csv(triplet_path, encoding="utf-8-sig")

    required_cols = ["anchor", "positive", "negative"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[ERROR] '{col}' 컬럼 누락 - generate_triplet_data.py 확인 필요")

    print(f"[DATA] 학습 Triplet 개수: {len(df):,}개")
    
    # 데이터 소스별 통계
    if "source" in df.columns:
        source_stats = df["source"].value_counts()
        print(f"[DATA] 데이터 소스별 통계:")
        for source, count in source_stats.items():
            print(f"  - {source}: {count:,}개 ({count/len(df)*100:.1f}%)")

    train_samples = [
        InputExample(texts=[row["anchor"], row["positive"], row["negative"]])
        for _, row in df.iterrows()
    ]

    # ============================================
    # Step 3: DataLoader & Loss 세팅
    # ============================================
    train_dataloader = DataLoader(
        train_samples,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=(device == "cuda")
    )

    warmup_steps = int(len(train_dataloader) * config.WARMUP_RATIO)
    train_loss = losses.TripletLoss(model)

    # ============================================
    # Step 4: 모델 학습 시작
    # ============================================
    print("\n==============================")
    print("모델 학습 시작")
    print("==============================")
    print("시작 시간:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    feedback_cnt = count_new_feedback()

    # epochs가 지정되지 않으면 config의 EPOCH_CONFIG를 사용하여 epoch 결정
    if epochs is None:
        if feedback_cnt == 0:
            epochs = config.EPOCH_CONFIG[0]
        elif feedback_cnt < 10:
            epochs = config.EPOCH_CONFIG[10]
        elif feedback_cnt < 50:
            epochs = config.EPOCH_CONFIG[50]
        else:
            epochs = config.EPOCH_CONFIG["default"]

    print(f"[TRAIN] feedback_cnt={feedback_cnt}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Warmup steps: {warmup_steps} ({config.WARMUP_RATIO*100}% of total steps)")
    print(f"Total training steps: {len(train_dataloader) * epochs}")
    print("==============================\n")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True,
        output_path=trained_model_path
    )

    # ============================================
    # Step 5: 최종 모델 저장
    # ============================================
    print("\n모델 저장 중…")

    try:
        os.makedirs(trained_model_path, exist_ok=True)
        model.save(trained_model_path)
        print(f"모델 저장 완료 → {trained_model_path}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] 모델 저장 실패: {e}")

    print("종료 시간:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("==============================")
    print("완료!")
    print(f"\n[INFO] 모델이 업데이트되었습니다.")
    print(f"[INFO] 다음 재학습 시 이 모델을 기반으로 추가 학습됩니다.")
    print(f"[INFO] 피드백이 쌓일수록 모델이 더 정확해집니다.")


# ============================================
# Entry point (명령줄 실행용)
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=None, help="배치 크기 (기본값: config.BATCH_SIZE)")
    parser.add_argument("--learning_rate", type=float, default=None, help="학습률 (기본값: config.LEARNING_RATE)")
    parser.add_argument("--triplet_path", type=str, default=str(config.TRIPLET_TRAINING_DATA_CSV),
                        help="Triplet 학습 데이터 경로")
    parser.add_argument("--model_output", type=str, default=str(config.TRAINED_MODEL_PATH),
                        help="모델 저장 경로")
    args = parser.parse_args()

    train_triplet_model(
        triplet_path=args.triplet_path,
        trained_model_path=args.model_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
