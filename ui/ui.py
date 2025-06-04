"""
ui.py
-----

GUI 기반의 자재 카테고리 분류
사용자는 자재명을 입력하고 추천 카테고리를 조회할 수 있으며,
선택된 카테고리에 대해 피드백 저장 및 모델 재학습

1. 사용자 입력 UI (자재명 입력, 언어 선택)
2. 분류 결과 Top-N 카테고리 출력
3. 선택된 카테고리에 대한 피드백 저장 (user_feedback.csv)
4. 추천 결과 없을 경우 직접 카테고리 제안
5. 모델 재학습 트리거 (기존 학습 파이프라인 호출)

"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import scripts.main

class MaterialCategoryApp:
    def __init__(self, root, classify_callback):
        self.root = root
        self.root.title("자재카테고리 분류 프로그램")
        self.root.configure(bg="white")  # 배경색 통일

        # 폰트 설정
        default_font = ("Segoe UI", 10)
        button_font = ("Segoe UI", 10, "bold")

        # 최상단: 재학습 버튼
        self.retrain_button = tk.Button(
            self.root, text="재학습 시작", command=self.retrain_model,
            bg="#2196F3", fg="white", font=button_font, width=15
        )
        self.retrain_button.pack(anchor="w", padx=10, pady=10)

        # 자재명 입력 + 분류 버튼
        self.input_frame = tk.Frame(self.root, bg="white")
        self.input_frame.pack(pady=5)

        self.material_name_label = tk.Label(
            self.input_frame, text="자재명 입력:", font=default_font, bg="white"
        )
        self.material_name_label.pack(side=tk.LEFT, padx=(0, 5))

        self.material_name_entry = tk.Entry(
            self.input_frame, width=50, font=default_font
        )
        self.material_name_entry.pack(side=tk.LEFT)

        self.classify_button = tk.Button(
            self.input_frame, text="분류하기", command=self.classify_material,
            bg="#4CAF50", fg="white", font=button_font, width=12
        )
        self.classify_button.pack(side=tk.LEFT, padx=(5, 0))

        # 언어 선택
        self.language_frame = tk.Frame(self.root, bg="white")
        self.language_frame.pack(pady=5)
        self.language_var = tk.StringVar(value="ENG")
        tk.Label(self.language_frame, text="언어 선택:", font=default_font, bg="white").pack(side=tk.LEFT)
        tk.Radiobutton(self.language_frame, text="영어", variable=self.language_var, value="ENG", font=default_font, bg="white").pack(side=tk.LEFT)
        tk.Radiobutton(self.language_frame, text="한국어", variable=self.language_var, value="KOR", font=default_font, bg="white").pack(side=tk.LEFT)

        # 추천 카테고리 리스트
        self.category_tree_label = tk.Label(
            self.root, text="추천 카테고리 리스트:", font=default_font, bg="white"
        )
        self.category_tree_label.pack()
        self.category_tree = ttk.Treeview(
            self.root, columns=("Category 1", "Category 2", "Category 3", "Score"), show="headings"
        )
        self.category_tree.heading("Category 1", text="카테고리 1")
        self.category_tree.heading("Category 2", text="카테고리 2")
        self.category_tree.heading("Category 3", text="카테고리 3")
        self.category_tree.heading("Score", text="유사도")
        self.category_tree.pack(pady=10)

        # 선택 + 적합 대상 없음 + 종료 버튼
        self.button_frame = tk.Frame(self.root, bg="white")
        self.button_frame.pack(pady=5)

        self.select_button = tk.Button(
            self.button_frame, text="선택", command=self.select_category,
            bg="#4CAF50", fg="white", font=button_font, width=15
        )
        self.select_button.pack(side=tk.LEFT, padx=10)

        self.no_match_button = tk.Button(
            self.button_frame, text="적합 대상 없음", command=self.no_match,
            bg="#03A9F4", fg="white", font=button_font, width=15
        )
        self.no_match_button.pack(side=tk.LEFT, padx=10)

        self.exit_button = tk.Button(
            self.button_frame, text="종료", command=self.root.quit,
            bg="#f44336", fg="white", font=button_font, width=15
        )
        self.exit_button.pack(side=tk.LEFT, padx=10)

        # 결과 출력
        self.result_label = tk.Label(
            self.root, text="결과가 여기에 표시됩니다.", justify="left", anchor="w",
            font=default_font, bg="white"
        )
        self.result_label.pack(pady=10)

        self.classify_callback = classify_callback
        self.last_result = []

    def classify_material(self):
        material_name = self.material_name_entry.get().strip()
        language = self.language_var.get()

        if not material_name:
            messagebox.showerror("입력 오류", "자재명을 입력하세요.")
            return

        result = self.classify_callback(material_name, language)

        self.last_result = result
        for row in self.category_tree.get_children():
            self.category_tree.delete(row)

        if result:
            for category in result:
                self.category_tree.insert("", "end", values=(
                    category['Level 1'], category['Level 2'], category['Level 3'], f"{category['Score']:.3f}"))
        else:
            self.result_label.config(text="결과를 찾을 수 없습니다.")

    def select_category(self):
        selected_item = self.category_tree.selection()
        if selected_item:
            selected_index = self.category_tree.index(selected_item)
            selected_category = self.category_tree.item(selected_item, "values")
            self.result_label.config(text=f"선택된 카테고리: {selected_category}")

            if self.last_result and selected_index < len(self.last_result):
                selected = self.last_result[selected_index]
                try:
                    with open("data/user_feedback.csv", "a", encoding="utf-8") as f:
                        f.write(f"{self.material_name_entry.get()},{self.language_var.get()},"
                                f"{selected['Level 1']},{selected['Level 2']},{selected['Level 3']}\n")
                    messagebox.showinfo("저장 완료", "사용자 피드백이 저장되었습니다.")
                except Exception as e:
                    messagebox.showerror("저장 오류", f"피드백 저장 중 오류 발생:\n{e}")
        else:
            messagebox.showerror("선택 오류", "카테고리를 선택해주세요.")

    def no_match(self):
        new_window = tk.Toplevel(self.root)
        new_window.title("새 카테고리 제안")

        default_font = ("Segoe UI", 10)

        material_name = self.material_name_entry.get().strip()

        tk.Label(new_window, text="자재명:", font=default_font).pack()
        entry_material = tk.Entry(new_window, width=50, font=default_font)
        entry_material.pack()
        entry_material.insert(0, material_name)
        entry_material.config(state="readonly")

        tk.Label(new_window, text="추가할 카테고리:", font=default_font).pack()
        entry_category = tk.Entry(new_window, width=50, font=default_font)
        entry_category.pack()

        def send_category():
            suggested_category = entry_category.get().strip()
            if suggested_category:
                print(f"새 제안 - 자재명: {material_name}, 제안 카테고리: {suggested_category}")
                messagebox.showinfo("전송 완료", "새 카테고리 제안이 전송되었습니다.")
                new_window.destroy()
            else:
                messagebox.showerror("입력 오류", "추가할 카테고리를 입력해주세요.")

        tk.Button(new_window, text="전송", command=send_category, bg="#2196F3", fg="white", font=("Segoe UI", 10, "bold")).pack(pady=10)

    def retrain_model(self):
        try:
            subprocess.run(["python", "-m", "train.generate_training_data"], check=True)
            subprocess.run(["python", "-m", "train.train_model"], check=True)
            subprocess.run(["python", "-m", "train.build_faiss_index"], check=True)
            messagebox.showinfo("성공", "모델 재학습이 완료되었습니다.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("오류", f"재학습 중 오류 발생:\n{e}")

def run_ui(classify_callback):
    root = tk.Tk()
    app = MaterialCategoryApp(root, classify_callback)
    root.mainloop()

if __name__ == "__main__":
    run_ui(scripts.main.classify_material)
# 안녕하세요 Test 입니다.