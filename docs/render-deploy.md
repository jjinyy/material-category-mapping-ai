# Render 배포 가이드

[Render 대시보드](https://dashboard.render.com)에서 이 저장소를 연결하면 Streamlit 앱을 외부에 노출할 수 있습니다.

## 1. Render에서 서비스 만들기

1. **https://dashboard.render.com** 접속 후 로그인
2. **New +** → **Web Service** 선택
3. **Connect a repository**에서 GitHub 계정 연결 후 `categoryMapping` 저장소 선택
4. 아래처럼 설정 (또는 저장소의 `render.yaml` 사용 시 일부 자동 채워짐)
   - **Name**: `category-mapping` (원하는 이름 가능)
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Plan**: Free

5. **Create Web Service** 클릭 후 빌드·배포 대기

## 2. 분류 기능이 동작하려면 (FAISS/모델)

앱은 올라가지만, **자재 분류**를 하려면 학습된 모델과 FAISS 인덱스가 필요합니다.  
현재 `models/` 폴더는 `.gitignore`에 있어 Git에는 없습니다.

**방법 A – 모델을 저장소에 포함 (권장)**  
로컬에서 한 번 학습한 뒤, 생성된 파일만 커밋합니다.

- `models/trained_model/` (학습된 SentenceTransformer)
- `models/faiss_KR_index.bin`, `models/faiss_KR_mapping.json`  
  (그리고 영어 사용 시 `faiss_EN_*`)

`.gitignore`에서 위 경로를 일시적으로 제거하거나, 별도 브랜치/스토리지에 두고 빌드 시 복사하는 방식으로 포함할 수 있습니다.

**방법 B – Render 빌드 시 생성**  
Build Command 뒤에 `admin/retrain.py` 또는 FAISS만 생성하는 스크립트를 실행하도록 넣는 방법입니다.  
(실행 시간이 길어지고 Free 플랜 제한에 걸릴 수 있음)

## 3. 배포 후

- 배포가 끝나면 **Dashboard**에 표시된 URL로 접속
- GitHub `main`에 push하면 Render가 자동으로 다시 배포합니다.
