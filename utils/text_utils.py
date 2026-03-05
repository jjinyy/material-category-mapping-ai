"""
텍스트 처리 유틸리티 함수들
언어 감지, 텍스트 정리 등 공통 기능 제공
"""

import re
import unicodedata
from typing import Literal


# ============================================
# 언어 감지
# ============================================
def detect_language(text: str, return_format: Literal["code", "full"] = "code") -> str:
    """
    텍스트에서 언어를 감지합니다.
    
    Args:
        text: 감지할 텍스트
        return_format: 반환 형식
            - "code": "KR" 또는 "EN" 반환 (기본값)
            - "full": "kor" 또는 "eng" 반환
    
    Returns:
        str: 감지된 언어 코드
    """
    text = str(text)
    
    if re.search(r"[가-힣]", text):
        return "KR" if return_format == "code" else "kor"
    
    if re.search(r"[a-zA-Z]", text):
        return "EN" if return_format == "code" else "eng"
    
    # 기본값: 영문으로 간주
    return "EN" if return_format == "code" else "eng"


# ============================================
# 텍스트 정리 함수들
# ============================================
def preprocess_material_name(text: str) -> str:
    """
    자재명을 전처리합니다.
    - "/" 앞부분만 사용
    - 괄호 내용 제거
    - "호" 단위 제거
    
    Args:
        text: 원본 자재명
    
    Returns:
        str: 전처리된 자재명
    """
    text = text.split("/")[0]
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\d+호", "", text)
    return text.strip()


def cleanse_text(text: str, mode: Literal["strict", "normal"] = "normal") -> str:
    """
    텍스트를 정리합니다 (유니코드 정규화, 특수문자 제거 등).
    
    Args:
        text: 원본 텍스트
        mode: 정리 모드
            - "normal": 기본 정리 (괄호 제거, 소문자 변환)
            - "strict": 엄격한 정리 (한글/영문/숫자만 허용)
    
    Returns:
        str: 정리된 텍스트
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\([^)]*\)", "", text)
    
    if mode == "strict":
        # 한글, 영문, 숫자만 허용
        text = re.sub(r"[^\w\s가-힣一-龥ぁ-ゔァ-ヴー々〆đĐơƯưôơêâăÁáÀàẢảÃãẠạĂăẮắẰằẲẵẶặÂầấẩẫậ]", " ", text)
    else:
        # 기본: 괄호만 제거하고 나머지는 유지
        text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_name(name: str) -> str:
    """
    자재명을 정리합니다 (generate_triplet_data.py용).
    
    Args:
        name: 원본 자재명
    
    Returns:
        str: 정리된 자재명
    """
    name = str(name).strip().lower()
    name = name.replace("/", " ")
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\d+[\d\.\*xX]+\d+", " ", name)
    name = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name




