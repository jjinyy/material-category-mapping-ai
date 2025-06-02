import pandas as pd

# 사용자 피드백 파일
feedback_file = "user_feedback.csv"
output_file = "training_data.csv"

# 언어별로 label 만들기
def generate_training_data():
    df = pd.read_csv(feedback_file)
    training_rows = []

    for _, row in df.iterrows():
        material = row["material_name"]
        lang = row["language"]

        if lang == "ENG":
            label = f"{row['Level 1_ENG']} > {row['Level 2_ENG']} > {row['Level 3_ENG']}"
        else:
            label = f"{row['Level 1_KOR']} > {row['Level 2_KOR']} > {row['Level 3_KOR']}"

        training_rows.append({
            "sentence1": material,
            "sentence2": label
        })

    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(output_file, index=False)
    print(f"훈련 데이터 저장 완료: {output_file}")

if __name__ == "__main__":
    generate_training_data()
