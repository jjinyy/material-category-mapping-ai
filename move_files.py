import os
import shutil

# 이동할 파일 목록 (source -> target directory)
file_moves = {
    "train_model.py": "train/",
    "generate_training_data.py": "train/",
    "init_train.py": "train/",
    "ui.py": "ui/",
    "category.csv": "data/",
    "training_data.csv": "data/",
    "user_feedback.csv": "data/",
    "faiss_index.bin": "models/",
    "faiss_mapping.json": "models/",
    "faiss_metadata.csv": "models/",
    "main.py": "scripts/",
    "test.py": "scripts/",
    "폴더구조.txt": "docs/"
}

# 디렉토리 이동
dirs_to_move = {
    "trained_model": "models/",
    "checkpoints": "models/"
}

# 이동 시작
for file, target_dir in file_moves.items():
    os.makedirs(target_dir, exist_ok=True)  # 디렉토리 없으면 만들기
    shutil.move(file, os.path.join(target_dir, file))
    print(f"Moved file {file} -> {target_dir}")

for folder, target_dir in dirs_to_move.items():
    os.makedirs(target_dir, exist_ok=True)
    shutil.move(folder, os.path.join(target_dir, folder))
    print(f"Moved folder {folder} -> {target_dir}")
