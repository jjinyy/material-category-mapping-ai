import streamlit as st
import pandas as pd
import os
import sys
import csv
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
# Helper Functions
# --------------------------------------
def save_feedback_to_csv(material_name, material_type, category_data, status="NEW"):
    """
    사용자 피드백을 CSV 파일에 저장합니다.
    
    Args:
        material_name: 자재명
        material_type: 자재 타입 (ROH1, ROH2)
        category_data: 카테고리 정보 딕셔너리
            - NEW 상태: {"L1": ..., "L2": ..., "L3": ..., "L4": ..., "CODE": ..., "Score": ..., "rank": ...}
            - PROPOSED 상태: {"new_category": ...} (L1~L4, CODE는 빈 문자열)
        status: 피드백 상태 ("NEW" 또는 "PROPOSED")
    
    Returns:
        bool: 저장 성공 여부
    """
    path = str(config.USER_FEEDBACK_CSV)
    config.ensure_dirs()
    exists = os.path.exists(path)
    
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            
            # 헤더가 없으면 추가
            if not exists:
                writer.writerow([
                    "material_name", "material_type", "lang",
                    "L1", "L2", "L3", "L4", "code",
                    "model_score", "rank",
                    "status", "created_at", "used_at"
                ])
            
            # 데이터 행 작성
            if status == "NEW":
                writer.writerow([
                    material_name,
                    material_type,
                    "",  # lang (generate에서 detect)
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
                ])
            elif status == "PROPOSED":
                writer.writerow([
                    material_name,
                    material_type,
                    "",  # lang (generate에서 detect)
                    "", "", "", "",  # L1~L4 없음
                    "",  # code 없음
                    0,   # model_score
                    0,   # rank
                    status,
                    datetime.now().isoformat(),
                    "",
                ])
        
        return True
    except Exception as e:
        st.error(f"피드백 저장 중 오류 발생: {e}")
        return False


# --------------------------------------
# Streamlit 기본 설정
# --------------------------------------
st.set_page_config(
    page_title="CJ 자재 카테고리 자동 분류",
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
.sub-title { font-size: 18px; color: #6B7280; margin-bottom: 18px; }

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
# Title
# --------------------------------------
st.markdown("<div class='main-title'>CJ 자재 카테고리 자동 분류 시스템</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI 기반으로 자재명을 분석하여 최적의 카테고리를 추천합니다.</div>", unsafe_allow_html=True)

# --------------------------------------
# Session State 초기화
# --------------------------------------
for k, v in {
    "results": [],
    "last_material": "",
    "last_type": "",
    "selected_row": None,
    "saved_category": None,  # 저장된 카테고리 정보
    "input_version": 0,
    "grid_version": 0,
    "feedback_saved": False,  # 피드백 저장 여부
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --------------------------------------
# Input Area
# --------------------------------------
with st.container():
    st.markdown("<div class='input-anchor'></div>", unsafe_allow_html=True)

    col_type, col_name, col_run, col_reset = st.columns(
        [1.3, 4.7, 1.6, 1.8], vertical_alignment="bottom"
    )

    with col_type:
        selected_type = st.selectbox("자재 타입", ["ROH1", "ROH2"], key="material_type")

    with col_name:
        material_name = st.text_input(
            "자재명",
            placeholder="예: 시트/하부PE시트(0.02*130*1000)*PM 또는 soybean refined oil",
            key=f"material_input_{st.session_state.input_version}",
        )

    with col_run:
        run_button = st.button("분류하기", type="primary", use_container_width=True)

    with col_reset:
        reset_button = st.button("새 자재명 입력하기", use_container_width=True)

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
        '<div class="notice-box">먼저 자재명을 입력하고 분류하기 버튼을 눌러주세요.</div>',
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
        status="NEW"
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
# New Category Proposal (PROPOSED feedback)
# --------------------------------------
st.markdown("---")
st.markdown("### 적합한 카테고리가 없나요?")

with st.expander("➕ 새 카테고리 제안하기"):
    new_category = st.text_input(
        "제안할 카테고리 (자유롭게 작성해주세요)",
        placeholder="예: 농산물 > 고추가공 > 고춧가루(유기농)",
        key="new_category_name",
    )

    propose_button = st.button(
        "제안 전송",
        disabled=has_selection,   # 선택돼 있으면 제안 불가
        key="propose_category"
    )

    if propose_button:
        if not new_category.strip():
            st.error("카테고리를 입력해주세요.")
        else:
            success = save_feedback_to_csv(
                material_name=st.session_state.get("last_material", ""),
                material_type=st.session_state.get("last_type", ""),
                category_data={"new_category": new_category},
                status="PROPOSED"
            )
            
            if success:
                st.success("새 카테고리 제안이 접수되었습니다. 감사합니다 🙏")
