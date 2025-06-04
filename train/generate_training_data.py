"""
generate_training_data.py
--------------------------

사용자 피드백(user_feedback.csv)을 읽어
모델 학습에 사용할 훈련 데이터(training_data.csv)생성

1. 사용자 피드백 데이터 로드 (data/user_feedback.csv)
2. 자재명과 선택된 카테고리 정보를 조합하여 훈련 샘플 생성
3. 훈련 데이터 CSV로 저장 (data/training_data.csv)

"""

import pandas as pd

# 파일 경로 설정
feedback_file = "data/user_feedback.csv"
output_file = "data/training_data.csv"

def generate_training_data():
    """
    사용자 피드백을 기반으로 학습용 데이터셋 생성
    """
    df = pd.read_csv(feedback_file)
    training_rows = []

    for _, row in df.iterrows():
        material = row["material_name"]
        # Level 정보는 언어와 무관하게 저장됨
        label = f"{row['Level 1']} > {row['Level 2']} > {row['Level 3']}"

        training_rows.append({
            "sentence1": material,
            "sentence2": label
        })

    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(output_file, index=False)
    print(f"훈련 데이터 저장 완료: {output_file}")

if __name__ == "__main__":
    generate_training_data()
