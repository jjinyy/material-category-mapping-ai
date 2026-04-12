import streamlit as st
import pandas as pd
import os
import sys
import csv
import hashlib
from datetime import datetime

from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode

# --------------------------------------
# config 모듈 import
# --------------------------------------
import config

# --------------------------------------
# scripts/main.py 연결
# --------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
import scripts.main as main


# --------------------------------------
# 로그인 (인증) Helper
# --------------------------------------
AUTH_SALT = "category_mapping_2025"


def _hash_password(password: str) -> str:
    return hashlib.sha256((AUTH_SALT + password).encode()).hexdigest()


def _load_users():
    """data/users.csv에서 username, password_hash 목록 로드."""
    path = str(config.USERS_CSV)
    if not os.path.exists(path):
        return []
    users = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)  # 헤더 스킵
            for row in reader:
                if len(row) >= 2:
                    users.append((row[0].strip(), row[1].strip()))
    except Exception:
        pass
    return users


def _verify_user(username: str, password: str) -> bool:
    users = _load_users()
    h = _hash_password(password)
    return any(u[0] == username and u[1] == h for u in users)


def _add_user(username: str, password: str) -> bool:
    """새 사용자 추가 (최초 가입용)."""
    path = str(config.USERS_CSV)
    config.ensure_dirs()
    exists = os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["username", "password_hash"])
            writer.writerow([username.strip(), _hash_password(password)])
        return True
    except Exception:
        return False


# --------------------------------------
# Helper Functions
# --------------------------------------
def _ensure_user_id_in_feedback_csv(path: str) -> None:
    """기존 user_feedback.csv에 user_id 컬럼이 없으면 맨 앞에 추가하여 마이그레이션."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            lines = f.readlines()
        if not lines:
            return
        header = lines[0].strip()
        if "user_id" in header.lower():
            return
        # 기존 헤더 앞에 user_id 추가
        new_header = "user_id," + header
        new_lines = [new_header + "\n"]
        for line in lines[1:]:
            stripped = line.rstrip("\n\r")
            if stripped:
                new_lines.append("," + stripped + "\n")  # 기존 행에는 user_id 빈 값
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.writelines(new_lines)
    except Exception:
        pass


def save_feedback_to_csv(material_name, material_type, category_data, status="NEW", user_id=""):
    """
    사용자 피드백을 CSV 파일에 저장합니다. 로그인한 user_id가 맨 앞 컬럼으로 기록됩니다.

    Args:
        material_name: 자재명
        material_type: 자재 타입 (ROH1, ROH2)
        category_data: 카테고리 정보 딕셔너리
        status: 피드백 상태 ("NEW" 또는 "PROPOSED")
        user_id: 로그인한 사용자 ID (계정으로 카운트용)
    """
    path = str(config.USER_FEEDBACK_CSV)
    config.ensure_dirs()
    _ensure_user_id_in_feedback_csv(path)
    exists = os.path.exists(path)
    header = [
        "user_id", "material_name", "material_type", "lang",
        "L1", "L2", "L3", "L4", "code",
        "model_score", "rank",
        "status", "created_at", "used_at", "new_category"
    ]
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(header)
            if status == "NEW":
                writer.writerow([
                    user_id or "",
                    material_name,
                    material_type,
                    "",
                    category_data.get("L1", ""),
                    category_data.get("L2", ""),
                    category_data.get("L3", ""),
                    category_data.get("L4", ""),
                    category_data.get("CODE", ""),
                    category_data.get("Score", 0),
                    category_data.get("rank", 0),
                    status,
                    datetime.now().isoformat(),
                    "",
                    "",
                ])
            elif status == "PROPOSED":
                writer.writerow([
                    user_id or "",
                    material_name,
                    material_type,
                    "",      # lang
                    "", "", "", "",   # L1~L4
                    "",      # code
                    0,       # model_score
                    0,       # rank
                    status,
                    datetime.now().isoformat(),
                    "",      # used_at
                    category_data.get("new_category", ""),
                ])
        return True
    except Exception as e:
        st.error(f"피드백 저장 중 오류 발생: {e}")
        return False


# --------------------------------------
# Streamlit 기본 설정
# --------------------------------------
st.set_page_config(
    page_title="자재 카테고리 자동 분류",
    layout="wide",
    page_icon="📦",
)

# --------------------------------------
# Global CSS
# --------------------------------------
st.markdown(
    """
<style>
.main-title { font-size: 34px; font-weight: 800; color: #1F2937; }
/* 메뉴 하단 설명 문구(부제/캡션)는 모두 동일한 크기/색으로 통일 */
.sub-title { font-size: 16px; color: #374151; margin-bottom: 18px; }
div[data-testid="stCaption"] { font-size: 16px !important; color: #374151 !important; }

.section-card:has(.input-anchor) {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 18px 18px 8px 18px;
    margin-bottom: 22px;
}

div.stButton > button {
    border-radius: 12px !important;
    height: 46px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

button[kind="primary"]{
    background-color: #1E3A8A !important;
    color: #fff !important;
}

button.secondary {
    background-color: #F3F4F6 !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
}

.notice-box {
    background-color: #EFF6FF;
    color: #1E40AF;
    padding: 14px 16px;
    border-radius: 12px;
    font-weight: 500;
    margin-top: 16px;
}
.highlight-box {
    background-color: #ECFDF5;
    padding: 14px 16px;
    border-radius: 12px;
    font-weight: 600;
    color: #065F46;
    margin: 14px 0 10px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------
# Session State 초기화
# --------------------------------------
for k, v in {
    "authenticated": False,
    "user_id": "",
    "view": "main",  # "main" | "all_categories" | "dashboard"
    "results": [],
    "last_material": "",
    "last_type": "",
    "selected_row": None,
    "saved_category": None,
    "input_version": 0,
    "grid_version": 0,
    "feedback_saved": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------------------
# 로그인 확인 — 미인증 시 로그인/최초 가입 화면만 표시
# --------------------------------------
if not st.session_state.get("authenticated", False):
    users = _load_users()
    # 최초 1명도 없으면 관리자 계정 생성
    if not users:
        st.markdown("### 최초 설정: 관리자 계정 만들기")
        st.caption("한 번만 설정하면 됩니다. 이후 이 계정으로 로그인합니다.")
        c1, c2 = st.columns(2)
        with c1:
            init_username = st.text_input("아이디", key="init_username")
        with c2:
            init_password = st.text_input("비밀번호", type="password", key="init_password")
        if st.button("계정 생성"):
            if init_username.strip() and init_password:
                if _add_user(init_username.strip(), init_password):
                    st.session_state["authenticated"] = True
                    st.session_state["user_id"] = init_username.strip()
                    st.success("계정이 생성되었습니다. 로그인되었습니다.")
                    st.rerun()
                else:
                    st.error("계정 생성에 실패했습니다.")
            else:
                st.warning("아이디와 비밀번호를 입력하세요.")
    else:
        st.markdown("### 로그인")
        login_user = st.text_input("아이디", key="login_user")
        login_pass = st.text_input("비밀번호", type="password", key="login_pass")
        if st.button("로그인"):
            if _verify_user(login_user.strip(), login_pass):
                st.session_state["authenticated"] = True
                st.session_state["user_id"] = login_user.strip()
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

        st.markdown("---")
        with st.expander("처음 오셨나요? 회원가입"):
            st.caption("새 계정을 만들면 로그인하여 피드백을 남길 수 있습니다.")
            reg_user = st.text_input("가입할 아이디", key="reg_username")
            reg_pass = st.text_input("비밀번호", type="password", key="reg_password")
            reg_pass2 = st.text_input("비밀번호 확인", type="password", key="reg_password_confirm")
            if st.button("가입하기", key="btn_register"):
                if not reg_user.strip():
                    st.error("아이디를 입력하세요.")
                elif not reg_pass:
                    st.error("비밀번호를 입력하세요.")
                elif reg_pass != reg_pass2:
                    st.error("비밀번호가 일치하지 않습니다.")
                else:
                    existing = _load_users()
                    if any(u[0] == reg_user.strip() for u in existing):
                        st.error("이미 사용 중인 아이디입니다.")
                    elif _add_user(reg_user.strip(), reg_pass):
                        st.session_state["authenticated"] = True
                        st.session_state["user_id"] = reg_user.strip()
                        st.success("가입되었습니다. 로그인된 상태로 이용 중입니다.")
                        st.rerun()
                    else:
                        st.error("가입에 실패했습니다.")
    st.stop()

# --------------------------------------
# 사이드바 (로그인 정보 + 뷰 전환 메뉴 + 로그아웃)
# --------------------------------------
with st.sidebar:
    st.caption(f"로그인: **{st.session_state.get('user_id', '')}**")
    if st.button("로그아웃"):
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = ""
        st.rerun()

    st.markdown("---")
    current_view = st.session_state.get("view", "main")

    # 세 개의 메뉴 버튼을 항상 표시 (현재 선택된 메뉴는 상태만 변경하지 않도록 동일 뷰 클릭 시에도 그대로 유지)
    if st.button("자재 분류", use_container_width=True):
        st.session_state["view"] = "main"
        st.rerun()
    if st.button("전체 카테고리 보기", use_container_width=True):
        st.session_state["view"] = "all_categories"
        st.rerun()
    if st.button("기여 대시보드", use_container_width=True):
        st.session_state["view"] = "dashboard"
        st.rerun()

# --------------------------------------
# 전체 카테고리 보기 화면
# --------------------------------------
if st.session_state.get("view") == "all_categories":
    st.markdown("<div class='main-title'>전체 카테고리 목록</div>", unsafe_allow_html=True)
    try:
        df_cat = pd.read_csv(str(config.CATEGORY_CSV), encoding="utf-8-sig")
        # L1~L4 해당 언어 컬럼만 선택 (TYPE, CODE 포함)
        # 언어 선택은 캡션과 같은 라인에서 오른쪽에 배치
        lang_suffix_map = {"한글": "_KR", "영문": "_EN"}
        # 임시로 한글 기준으로 한번 구성해두고, 실제 언어 선택 후 다시 세팅
        default_suffix = "_KR"
        level_cols = [f"L{i}{default_suffix}" for i in range(1, 5) if f"L{i}{default_suffix}" in df_cat.columns]
        display_cols = ["TYPE", "CODE"] + level_cols
        df_display = df_cat[display_cols].copy()

        # 상단 라인: 왼쪽 캡션, 오른쪽 언어 선택 (표의 오른쪽과 정렬되도록 오른쪽 컬럼을 작게 사용)
        c1, c2 = st.columns([7, 1])
        with c2:
            st.markdown("<div style='text-align:right;'>", unsafe_allow_html=True)
            lang_choice = st.radio(
                "표시 언어",
                ["한글", "영문"],
                horizontal=True,
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        suffix = lang_suffix_map.get(lang_choice, "_KR")

        # 선택된 언어 기준으로 다시 컬럼 구성
        level_cols = [f"L{i}{suffix}" for i in range(1, 5) if f"L{i}{suffix}" in df_cat.columns]
        display_cols = ["TYPE", "CODE"] + level_cols
        df_display = df_cat[display_cols].copy()
        df_display.columns = ["타입", "코드", "L1", "L2", "L3", "L4"][: len(display_cols)]

        with c1:
            st.markdown(
                f"<div class='sub-title'>총 {len(df_display):,}개 카테고리 "
                f"({lang_choice} 기준)</div>",
                unsafe_allow_html=True,
            )
        styled_cat = df_display.style.set_properties(**{"text-align": "center"})
        st.dataframe(styled_cat, use_container_width=True, height=600)
    except Exception as e:
        st.error(f"카테고리 목록을 불러올 수 없습니다: {e}")
    st.stop()

# --------------------------------------
# 기여 대시보드 화면
# --------------------------------------
if st.session_state.get("view") == "dashboard":
    st.markdown("<div class='main-title'>기여 대시보드</div>", unsafe_allow_html=True)
    try:
        df_fb = pd.read_csv(
            str(config.USER_FEEDBACK_CSV),
            encoding="utf-8-sig",
            on_bad_lines="skip",  # 필드 개수가 맞지 않는 문제 행은 건너뜀
        )
        if df_fb.empty:
            st.info("아직 저장된 피드백이 없습니다.")
            st.stop()
        if "user_id" not in df_fb.columns:
            st.warning("user_id 컬럼이 없어 기여자를 구분할 수 없습니다.")
            st.dataframe(df_fb, use_container_width=True, height=500)
            st.stop()

        df_fb["user_id"] = df_fb["user_id"].fillna("").replace("", "unknown")
        df_fb["created_at"] = pd.to_datetime(df_fb["created_at"], errors="coerce")
        df_fb = df_fb.dropna(subset=["created_at"])

        # 기여 대시보드는 재학습(USED) 여부와 관계없이 모든 피드백을 노출 (retrain 후에도 기여 이력 유지)
        # (status 필터 제거: NEW, PROPOSED, USED 전부 기여로 집계)

        # 타이틀 바로 아래 설명 텍스트
        st.markdown(
            "<div class='sub-title'>각 사용자별 기간별 피드백 기여 현황입니다.</div>",
            unsafe_allow_html=True,
        )

        # --------------------------------------
        # 오늘 기준 Daily / Weekly / Monthly TOP 3 요약 (컬러 카드)
        # --------------------------------------
        today_ts = pd.Timestamp.now().normalize()
        today_date = today_ts.date()

        df_daily = df_fb[df_fb["created_at"].dt.date == today_date]
        this_week = today_ts.to_period("W")
        df_week = df_fb[df_fb["created_at"].dt.to_period("W") == this_week]
        this_month = today_ts.to_period("M")
        df_month = df_fb[df_fb["created_at"].dt.to_period("M") == this_month]

        col_d, col_w, col_m = st.columns(3)
        summary_sets = [
            ("Daily", df_daily, col_d),
            ("Weekly", df_week, col_w),
            ("Monthly", df_month, col_m),
        ]

        card_colors = {
            "Daily": "#DBEAFE",   # 연한 파랑
            "Weekly": "#FEF3C7",  # 연한 노랑
            "Monthly": "#ECFDF3", # 연한 초록
        }

        for label, df_src, col in summary_sets:
            with col:
                bg = card_colors.get(label, "#F3F4F6")
                st.markdown(
                    f"<div class='sub-title' style='margin-bottom:4px;'>{label} TOP 3</div>",
                    unsafe_allow_html=True,
                )
                if df_src.empty:
                    st.markdown(
                        f"<div style='background:{bg}; border-radius:12px; padding:10px 12px; color:#4B5563; text-align:center;'>데이터 없음</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    top = (
                        df_src.groupby("user_id")["status"]
                        .count()
                        .reset_index()
                        .rename(columns={"user_id": "사용자", "status": "피드백 수"})
                        .sort_values("피드백 수", ascending=False)
                        .head(3)
                    )
                    # 순위/메달 컬럼 추가
                    medals = ["🥇", "🥈", "🥉"]
                    top = top.reset_index(drop=True)
                    top.index = top.index + 1
                    top.insert(0, "순위", [medals[i] if i < len(medals) else "" for i in range(len(top))])

                    styled_top = top.style.set_properties(**{"text-align": "center"}).set_table_styles(
                        [
                            {
                                "selector": "th",
                                "props": [
                                    ("background-color", bg),
                                    ("font-weight", "600"),
                                    ("border", "none"),
                                    ("text-align", "center"),
                                ],
                            },
                            {
                                "selector": "td",
                                "props": [
                                    ("border", "none"),
                                    ("padding", "6px 8px"),
                                ],
                            },
                            {
                                "selector": "tbody tr:nth-child(odd)",
                                "props": [("background-color", "#F9FAFB")],
                            },
                        ]
                    )
                    st.table(styled_top)

        # 설명 텍스트 아래: 왼쪽은 비워 두고, 오른쪽에 집계 단위 선택 (Daily/Weekly/Monthly)
        col_desc, col_period = st.columns([7, 1])
        with col_desc:
            st.empty()
        with col_period:
            period_label = st.selectbox(
                "집계 단위",
                options=["Daily", "Weekly", "Monthly"],
                index=0,
            )

        if period_label == "Daily":
            df_fb["period"] = df_fb["created_at"].dt.date
            period_name = "일자"
        elif period_label == "Weekly":
            df_fb["period"] = df_fb["created_at"].dt.to_period("W").dt.start_time.dt.date
            period_name = "주(시작일)"
        else:
            df_fb["period"] = df_fb["created_at"].dt.to_period("M").dt.start_time.dt.date
            period_name = "월(1일 기준)"

        grouped = (
            df_fb.groupby(["user_id", "period"])["status"]
            .agg(
                total="count",
                new=lambda s: (s == "NEW").sum(),
                proposed=lambda s: (s == "PROPOSED").sum(),
                used=lambda s: (s == "USED").sum(),
            )
            .reset_index()
        )

        grouped = grouped.rename(
            columns={
                "user_id": "사용자",
                "period": period_name,
                "total": "총 피드백 수",
                "new": "NEW 개수",
                "proposed": "PROPOSED 개수",
                "used": "USED(재학습반영) 개수",
            }
        )
        grouped = grouped.sort_values(by=[period_name, "총 피드백 수"], ascending=[False, False])
        # 인덱스 재정렬 (1부터 시작하는 연속 번호) → 번호가 15, 6처럼 튀는 현상 방지
        grouped = grouped.reset_index(drop=True)
        grouped.index = grouped.index + 1

        styled = grouped.style.set_properties(**{"text-align": "center"})
        st.dataframe(styled, use_container_width=True, height=600)
    except FileNotFoundError:
        st.info("user_feedback.csv 파일이 아직 생성되지 않았습니다. 먼저 피드백을 남겨주세요.")
    except Exception as e:
        st.error(f"대시보드를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# --------------------------------------
# Input Area (폼 사용 → 자재명 입력 후 Enter로도 분류 실행)
# (메인 분류 화면에서만 Title + 입력 영역 표시)
# --------------------------------------
if st.session_state.get("view", "main") == "main":
    st.markdown("<div class='main-title'>자재 카테고리 자동 분류 시스템</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>AI 기반으로 자재명을 분석하여 최적의 카테고리를 추천합니다.</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='input-anchor'></div>", unsafe_allow_html=True)

    with st.form("classify_form"):
        col_type, col_name, col_btns = st.columns([1.3, 4.7, 2.2], vertical_alignment="bottom")
        with col_type:
            selected_type = st.selectbox("자재 타입", ["ROH1", "ROH2"], key="material_type")
        with col_name:
            material_name = st.text_input(
                "자재명",
                placeholder="예: 시트/하부PE시트(0.02*130*1000)*PM 또는 soybean refined oil (입력 후 Enter)",
                key=f"material_input_{st.session_state.input_version}",
            )
        with col_btns:
            btn_left, btn_right = st.columns(2)
            with btn_left:
                run_button = st.form_submit_button("분류하기")
            with btn_right:
                reset_button = st.form_submit_button("새 자재명 입력하기")

# 버튼 스타일
st.markdown(
    """
<script>
document.querySelectorAll('button').forEach(b=>{
    if(b.innerText.includes('새 자재명')) b.classList.add('secondary');
})
</script>
""",
    unsafe_allow_html=True,
)

# --------------------------------------
# Reset
# --------------------------------------
if reset_button:
    st.session_state.update({
        "results": [],
        "last_material": "",
        "last_type": "",
        "selected_row": None,
        "saved_category": None,
        "feedback_saved": False,
        "input_version": st.session_state["input_version"] + 1,
        "grid_version": st.session_state["grid_version"] + 1,
    })
    # 전체 카테고리 직접 선택 관련 상태도 초기화
    for key in ["direct_category_search", "direct_category_select"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --------------------------------------
# Run Classification
# --------------------------------------
# 자재명이나 타입이 변경되었는지 확인 (분류하기 버튼 클릭 전에)
last_material = st.session_state.get("last_material", "")
last_type = st.session_state.get("last_type", "")
material_changed = (material_name != last_material) or (selected_type != last_type)

# 자재명이나 타입이 변경되었으면 이전 결과 초기화
if material_changed and material_name.strip():
    st.session_state.update({
        "selected_row": None,
        "saved_category": None,
        "feedback_saved": False,
    })

if run_button:
    if not material_name.strip():
        st.error("자재명을 입력하세요.")
    else:
        try:
            with st.spinner("AI가 자재명을 분석 중입니다..."):
                results = main.classify_material(
                    material_name=material_name,
                    selected_type=selected_type,
                )

            # 분류 시 항상 완전히 초기화
            st.session_state.update({
                "results": results,
                "last_material": material_name,
                "last_type": selected_type,
                "selected_row": None,  # 새 분류 시 선택 초기화
                "saved_category": None,
                "feedback_saved": False,  # 새 분류 시 피드백 저장 상태 초기화
                "grid_version": st.session_state.get("grid_version", 0) + 1,  # 그리드 강제 새로고침
            })
            # 새 분류 시 전체 카테고리 직접 선택 상태도 초기화
            for key in ["direct_category_search", "direct_category_select"]:
                if key in st.session_state:
                    del st.session_state[key]
        except FileNotFoundError as e:
            st.error(f"필수 파일을 찾을 수 없습니다.\n\n{str(e)}\n\nFAISS 인덱스를 먼저 생성해주세요.")
        except ValueError as e:
            st.error(f"입력값 오류: {str(e)}")
        except RuntimeError as e:
            st.error(f"시스템 오류: {str(e)}\n\n관리자에게 문의해주세요.")
        except Exception as e:
            st.error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
            st.exception(e)  # 디버깅용 상세 에러 표시

results = st.session_state["results"]

# --------------------------------------
# Results
# --------------------------------------
if not results:
    st.markdown(
        '<div class="notice-box">자재명을 입력한 뒤 Enter를 누르거나 분류하기 버튼을 눌러주세요.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown("### AI 추천 결과")
st.caption("점수가 높을수록 입력한 자재명과 의미적으로 유사한 카테고리입니다.")

df = pd.DataFrame(results)
df["rank"] = df["Score"].rank(ascending=False, method="first").astype(int)

df_display = df[["CODE", "TYPE", "L1", "L2", "L3", "L4", "Score", "rank"]].copy()
df_display["Score"] = (df_display["Score"] * 100).round(2)

# --------------------------------------
# AgGrid 설정 (🔥 JsCode 에러 해결 핵심)
# --------------------------------------
score_style = JsCode(
    """
    function(params) {
        if (params.value >= 90) return {color:'white', backgroundColor:'#047857'};
        if (params.value >= 70) return {color:'black', backgroundColor:'#FEF08A'};
        return {color:'white', backgroundColor:'#B91C1C'};
    }
    """
)

gb = GridOptionsBuilder.from_dataframe(df_display)
gb.configure_grid_options(
    rowSelection="single",
    suppressRowClickSelection=False,  # 행 클릭으로 선택 활성화
    allowUnsafeJsCode=True,
)
gb.configure_selection(
    selection_mode="single", 
    use_checkbox=False,
)
gb.configure_column("Score", headerName="Score(%)", cellStyle=score_style)

grid_response = AgGrid(
    df_display,
    gridOptions=gb.build(),
    update_on=["selection_changed", "cellClicked"],  # 셀 클릭도 감지
    theme="material",
    height=420,
    allow_unsafe_jscode=True,
    key=f"grid_{st.session_state['grid_version']}",
    reload_data=False,
    enable_enterprise_modules=False,
)

# 선택된 행 가져오기 - 사용자가 실제로 클릭한 경우만
selected_rows = None
selected_data = None
user_clicked = False  # 사용자가 실제로 클릭했는지 여부

# event_data에서 사용자 클릭 확인 (가장 확실한 방법)
try:
    if hasattr(grid_response, 'event_data') and grid_response.event_data:
        event_data = grid_response.event_data
        if isinstance(event_data, dict):
            # 사용자가 실제로 클릭한 경우만
            event_type = event_data.get('type', '')
            if event_type in ['selectionChanged', 'cellClicked', 'rowClicked']:
                user_clicked = True
except:
    pass

# 사용자가 클릭한 경우에만 selected_rows 확인
if user_clicked:
    # 방법 1: selected_rows 직접 접근
    try:
        if hasattr(grid_response, 'selected_rows'):
            selected_rows = grid_response.selected_rows
        if hasattr(grid_response, 'get'):
            selected_rows = grid_response.get("selected_rows", None) or selected_rows
    except Exception as e:
        pass

    # 방법 2: selected_data 확인
    if selected_rows is None or (isinstance(selected_rows, list) and len(selected_rows) == 0):
        try:
            if hasattr(grid_response, 'selected_data'):
                selected_data = grid_response.selected_data
            if hasattr(grid_response, 'get'):
                selected_data = grid_response.get("selected_data", None) or selected_data
            
            # selected_data가 있으면 사용
            if selected_data is not None:
                if isinstance(selected_data, pd.DataFrame) and not selected_data.empty:
                    selected_rows = selected_data
                elif isinstance(selected_data, list) and len(selected_data) > 0:
                    selected_rows = selected_data
        except Exception as e:
            pass

    # 방법 3: event_data에서 선택 정보 확인
    if selected_rows is None:
        try:
            if hasattr(grid_response, 'event_data') and grid_response.event_data:
                event_data = grid_response.event_data
                if isinstance(event_data, dict):
                    if 'rowIndex' in event_data or 'row' in event_data:
                        row_idx = event_data.get('rowIndex')
                        if row_idx is None and 'row' in event_data:
                            row_idx = event_data['row'].get('rowIndex')
                        if row_idx is not None and 0 <= row_idx < len(df_display):
                            selected_rows = [df_display.iloc[row_idx].to_dict()]
        except Exception as e:
            pass

selected_row = None

if selected_rows is not None:
    # 리스트 형태
    if isinstance(selected_rows, list):
        if len(selected_rows) > 0:
            item = selected_rows[0]
            if isinstance(item, dict):
                selected_row = item.copy()
            elif isinstance(item, pd.Series):
                selected_row = item.to_dict()
            elif hasattr(item, '__dict__'):
                selected_row = dict(item.__dict__)
            else:
                try:
                    selected_row = dict(item)
                except:
                    pass
    # DataFrame 형태
    elif isinstance(selected_rows, pd.DataFrame):
        if not selected_rows.empty:
            selected_row = selected_rows.iloc[0].to_dict()
    # 딕셔너리 형태
    elif isinstance(selected_rows, dict):
        selected_row = selected_rows.copy()

# 선택된 행이 있으면 session state 업데이트
# 사용자가 실제로 클릭한 경우만 업데이트 (자동 선택 방지)
if selected_row is not None and user_clicked and not st.session_state.get("feedback_saved", False):
    # 원본 데이터에서 누락된 정보 보완
    code = selected_row.get('CODE')
    if code:
        original_row = df[df['CODE'] == code]
        if not original_row.empty:
            orig = original_row.iloc[0]
            selected_row['rank'] = orig.get('rank', selected_row.get('rank'))
            selected_row['Score'] = orig.get('Score', selected_row.get('Score'))
    
    st.session_state["selected_row"] = selected_row
elif not user_clicked:
    # 사용자가 클릭하지 않았으면 이전 선택 유지하지 않음
    # 새로 분류한 경우는 이미 selected_row가 None으로 초기화됨
    # 따라서 아무것도 하지 않음 (자동 선택 방지)
    pass


# --------------------------------------
# Save Feedback
# --------------------------------------
st.markdown("---")
st.markdown("### 선택한 카테고리")

row = st.session_state.get("selected_row")
saved_category = st.session_state.get("saved_category")
has_selection = row is not None and isinstance(row, dict)
feedback_saved = st.session_state.get("feedback_saved", False)

if feedback_saved and saved_category:
    # 피드백이 이미 저장된 경우 - 저장된 카테고리 정보 표시
    l4_display = f" > {saved_category['L4']}" if saved_category.get('L4') and str(saved_category['L4']) != 'nan' and str(saved_category['L4']).strip() else ""
    st.success("피드백이 저장되었습니다. 다음 재학습에 반영됩니다.")
    st.info(
        f"**저장된 카테고리:** {saved_category['L1']} > {saved_category['L2']} > {saved_category['L3']}{l4_display}  | CODE: `{saved_category['CODE']}`"
    )
    st.caption("새로운 자재명을 입력하거나 '새 자재명 입력하기' 버튼을 눌러주세요.")
elif not has_selection:
    st.info("표에서 하나의 카테고리를 선택해주세요.")
    st.caption("표의 행을 클릭하면 선택됩니다.")
else:
    # 선택된 카테고리 정보 표시
    l4_display = f" > {row['L4']}" if row.get('L4') and str(row['L4']) != 'nan' and str(row['L4']).strip() else ""
    st.success(
        f"**{row['L1']} > {row['L2']} > {row['L3']}{l4_display}**  | CODE: `{row['CODE']}`"
    )

save_button = st.button(
    "이 카테고리로 확정하기",
    type="primary",
    use_container_width=True,
    disabled=not has_selection or feedback_saved,  # 피드백 저장 후 비활성화
)

if save_button and row and not feedback_saved:
    success = save_feedback_to_csv(
        material_name=st.session_state["last_material"],
        material_type=st.session_state["last_type"],
        category_data={
            "L1": row["L1"],
            "L2": row["L2"],
            "L3": row["L3"],
            "L4": row["L4"],
            "CODE": row["CODE"],
            "Score": row["Score"],
            "rank": row["rank"],
        },
        status="NEW",
        user_id=st.session_state.get("user_id", ""),
    )
    
    if success:
        # 저장된 카테고리 정보 보관
        st.session_state["saved_category"] = {
            "L1": row["L1"],
            "L2": row["L2"],
            "L3": row["L3"],
            "L4": row.get("L4", ""),
            "CODE": row["CODE"],
        }
        st.session_state["feedback_saved"] = True
        st.session_state["selected_row"] = None  # 선택 해제
        st.rerun()  # 페이지 새로고침하여 상태 반영


# --------------------------------------
# Direct category select / New proposal
# --------------------------------------
st.markdown("---")
st.markdown("### 적합한 카테고리가 없나요?")

# 검색 시 Enter를 쳐도 접히지 않도록 항상 펼쳐진 상태 유지
with st.expander("➕ 다른 방식으로 카테고리 남기기", expanded=True):
    tab_direct, tab_propose = st.tabs(["전체 카테고리에서 직접 선택", "새 카테고리 텍스트로 제안"])

    # 1) 전체 카테고리에서 직접 선택해서 확정 (NEW 피드백으로 저장)
    with tab_direct:
        st.caption("추천 목록에 안 보이는 카테고리는 전체 카테고리 목록에서 직접 선택해 확정할 수 있습니다.")
        try:
            df_cat = pd.read_csv(str(config.CATEGORY_CSV), encoding="utf-8-sig")
            last_type = st.session_state.get("last_type", "")
            if last_type:
                df_cat_view = df_cat[df_cat["TYPE"] == last_type].copy()
            else:
                df_cat_view = df_cat.copy()

            search_kw = st.text_input(
                "카테고리 검색 (코드 / 한글 / 영문 일부 입력)",
                key="direct_category_search",
            ).strip()

            if search_kw:
                kw = search_kw.lower()
                mask = df_cat_view.apply(
                    lambda row: kw in " ".join([str(v) for v in row.values]).lower(),
                    axis=1,
                )
                df_cat_view = df_cat_view[mask]

            if df_cat_view.empty:
                st.info("검색 조건에 맞는 카테고리가 없습니다.")
            else:
                # 검색어에 영문이 포함되면 EN 컬럼, 아니면 기본적으로 KR 컬럼 사용
                use_en = bool(search_kw) and any("a" <= ch.lower() <= "z" for ch in search_kw if ch.isascii())
                l1_col = "L1_EN" if use_en and "L1_EN" in df_cat_view.columns else "L1_KR"
                l2_col = "L2_EN" if use_en and "L2_EN" in df_cat_view.columns else "L2_KR"
                l3_col = "L3_EN" if use_en and "L3_EN" in df_cat_view.columns else "L3_KR"
                l4_col = "L4_EN" if use_en and "L4_EN" in df_cat_view.columns else "L4_KR"

                # 선택 박스 (검색 결과 중에서 하나를 확실하게 선택)
                options = list(df_cat_view.index)

                def _format_cat(idx: int) -> str:
                    r = df_cat_view.loc[idx]
                    parts = [
                        str(r.get(l1_col, "")),
                        str(r.get(l2_col, "")),
                        str(r.get(l3_col, "")),
                        str(r.get(l4_col, "")),
                    ]
                    path = " > ".join(p for p in parts if p and p != "nan")
                    return f"{r.get('CODE', '')} | {path}".strip()

                selected_idx = st.selectbox(
                    "직접 확정할 카테고리를 선택하세요.",
                    options,
                    format_func=_format_cat,
                    key="direct_category_select",
                )

                if selected_idx is not None:
                    r_prev = df_cat_view.loc[selected_idx]
                    path_parts = [
                        str(r_prev.get(l1_col, "")),
                        str(r_prev.get(l2_col, "")),
                        str(r_prev.get(l3_col, "")),
                        str(r_prev.get(l4_col, "")),
                    ]
                    path_txt = " > ".join(p for p in path_parts if p and p != "nan")
                    st.caption(f"현재 선택된 카테고리: `{r_prev.get('CODE', '')}` | {path_txt}")

                direct_confirm = st.button(
                    "이 카테고리로 직접 확정하기",
                    type="primary",
                    use_container_width=True,
                    key="direct_confirm_button",
                )

                if direct_confirm and selected_idx is not None:
                    r = df_cat_view.loc[selected_idx]
                    success = save_feedback_to_csv(
                        material_name=st.session_state.get("last_material", ""),
                        material_type=st.session_state.get("last_type", ""),
                        category_data={
                            "L1": r.get("L1_KR", ""),
                            "L2": r.get("L2_KR", ""),
                            "L3": r.get("L3_KR", ""),
                            "L4": r.get("L4_KR", ""),
                            "CODE": r.get("CODE", ""),
                            "Score": 0,
                            "rank": 0,
                        },
                        status="NEW",
                        user_id=st.session_state.get("user_id", ""),
                    )
                    if success:
                        st.success("선택한 카테고리로 피드백이 저장되었습니다. 다음 재학습에 반영됩니다.")
        except Exception as e:
            st.error(f"카테고리 목록을 불러오는 중 오류가 발생했습니다: {e}")

    # 2) 완전히 새로운 카테고리 텍스트로 제안 (PROPOSED)
    with tab_propose:
        new_category = st.text_input(
            "제안할 카테고리 (자유롭게 작성해주세요)",
            placeholder="예: 농산물 > 고추가공 > 고춧가루(유기농)",
            key="new_category_name",
        )

        propose_button = st.button(
            "제안 전송",
            key="propose_category",
        )

        if propose_button:
            if not new_category.strip():
                st.error("카테고리를 입력해주세요.")
            else:
                success = save_feedback_to_csv(
                    material_name=st.session_state.get("last_material", ""),
                    material_type=st.session_state.get("last_type", ""),
                    category_data={"new_category": new_category},
                    status="PROPOSED",
                    user_id=st.session_state.get("user_id", ""),
                )

                if success:
                    st.success("새 카테고리 제안이 접수되었습니다. 감사합니다 🙏")
