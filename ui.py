import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import main  # 실제 분류 함수가 있는 파일 (임포트 이름에 맞게 조정 필요)

class MaterialCategoryApp:
    def __init__(self, root, classify_callback):
        self.root = root
        self.root.title("자재카테고리 분류 프로그램")

        # 언어 선택
        self.language_var = tk.StringVar(value="ENG")
        self.language_frame = tk.Frame(self.root)
        self.language_frame.pack(pady=5)
        tk.Label(self.language_frame, text="언어 선택:").pack(side=tk.LEFT)
        tk.Radiobutton(self.language_frame, text="영어", variable=self.language_var, value="ENG").pack(side=tk.LEFT)
        tk.Radiobutton(self.language_frame, text="한국어", variable=self.language_var, value="KOR").pack(side=tk.LEFT)

        # 자재명 입력
        self.material_name_label = tk.Label(self.root, text="자재명 입력:")
        self.material_name_label.pack()
        self.material_name_entry = tk.Entry(self.root, width=50)
        self.material_name_entry.pack()

        # 분류 버튼
        self.classify_button = tk.Button(self.root, text="분류하기", command=self.classify_material)
        self.classify_button.pack(pady=5)

        # 카테고리 출력
        self.category_tree_label = tk.Label(self.root, text="추천 카테고리 리스트:")
        self.category_tree_label.pack()
        self.category_tree = ttk.Treeview(self.root, columns=("Category 1", "Category 2", "Category 3", "Score"), show="headings")
        self.category_tree.heading("Category 1", text="카테고리 1")
        self.category_tree.heading("Category 2", text="카테고리 2")
        self.category_tree.heading("Category 3", text="카테고리 3")
        self.category_tree.heading("Score", text="유사도")
        self.category_tree.pack()

        # 선택 버튼
        self.select_button = tk.Button(self.root, text="선택", command=self.select_category)
        self.select_button.pack(pady=5)

        # 재학습 버튼
        self.retrain_button = tk.Button(self.root, text="재학습 시작", command=self.retrain_model)
        self.retrain_button.pack(pady=10)

        # 결과 출력
        self.result_label = tk.Label(self.root, text="결과가 여기에 표시됩니다.", justify="left", anchor="w")
        self.result_label.pack()

        self.classify_callback = classify_callback
        self.last_result = []  # 선택한 분류 정보 저장용

    def classify_material(self):
        material_name = self.material_name_entry.get().strip()
        language = self.language_var.get()

        if not material_name:
            messagebox.showerror("입력 오류", "자재명을 입력하세요.")
            return

        result = self.classify_callback(material_name, language)

        self.last_result = result  # 선택 저장
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

            # 선택된 결과 저장 (선택된 index를 기준으로 self.last_result에서 가져오기)
            if self.last_result and selected_index < len(self.last_result):
                selected = self.last_result[selected_index]
                try:
                    with open("user_feedback.csv", "a", encoding="utf-8") as f:
                        f.write(f"{self.material_name_entry.get()},{self.language_var.get()},"
                                f"{selected['Level 1']},{selected['Level 2']},{selected['Level 3']}\n")
                    messagebox.showinfo("저장 완료", "사용자 피드백이 저장되었습니다.")
                except Exception as e:
                    messagebox.showerror("저장 오류", f"피드백 저장 중 오류 발생:\n{e}")
        else:
            messagebox.showerror("선택 오류", "카테고리를 선택해주세요.")

    def retrain_model(self):
        try:
            subprocess.run(["python", "generate_training_data.py"], check=True)
            subprocess.run(["python", "train_model.py"], check=True)
            subprocess.run(["python", "build_faiss_index.py"], check=True)
            messagebox.showinfo("성공", "모델 재학습이 완료되었습니다.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("오류", f"재학습 중 오류 발생:\n{e}")

def run_ui(classify_callback):
    root = tk.Tk()
    app = MaterialCategoryApp(root, classify_callback)
    root.mainloop()

if __name__ == "__main__":
    run_ui(main.classify_material)
