"""
工具注册表模块
提供MCP工具注册和参数管理功能
"""

from typing import Dict, Any, Optional, List, Callable
from pydantic import Field
from typing import Annotated

from tools.decorators import with_file_support_decorator as econometric_tool


# 文件输入参数定义
FILE_INPUT_PARAMS = {
    "file_content": Annotated[
        Optional[str],
        Field(
            default=None,
            description="""文件内容字符串，支持CSV/JSON/TXT格式

使用说明:
- CSV: 逗号分隔值格式
- JSON: {"列名": [값1, 값2, ...]} 또는 [{"열명1": 값1, ...}, ...]
- TXT: 다양한 텍스트 형식
  * 단일 열 데이터: 각 줄에 하나의 값
  * 다중 열 데이터: 공백 또는 탭으로 구분
  * 키-값 쌍: 열명: 값1 값2 값3

주의 사항:
- base64 인코딩 지원하지 않음
- 파일 내용이 파일 경로보다 우선
- file_content와 file_path 모두 제공될 경우 file_content 사용
- 파일 형식 자동 감지 가능, 수동 지정 가능(.csv/.json/.txt)

예시:
CSV: "1,2\\n1.2,3.4\\n2.3,4.5"
JSON: "{\\"x\\": [1,2,3], \\"y\\": [4,5,6]}"
TXT: "1.5\\n2.3\\n3.1"
TXT: "x y\\n1 2\\n3 4"
TXT: "x: 1 2 3\\ny: 4 5 6"
"""
        )
    ],
    "file_format": Annotated[
        str,
        Field(
            default="auto",
            description="""
파일 형식 유형

옵션:
- "auto": 자동 감지 - 내용에 따라 형식 자동 판단
- "csv": CSV 형식 - 쉼표로 구분된 값
- "json": JSON 형식 - JavaScript 객체 표기법
- "txt": TXT 형식 - 순수 텍스트 형식

감지 순서:
1. JSON 형식 우선
2. CSV 형식
3. TXT 형식
4. 파일 확장자(.csv/.json/.txt)에 따라 판단

주의 사항:
- "auto" 사용 권장
- 자동 감지 실패 시 수동 지정 필요
- TXT 형식은 다양한 구분자 지원"""
        )
    ]
}


class ToolConfig:
    """도구 구성 클래스"""
    
    def __init__(
        self,
        name: str,
        impl_func: Callable,
        tool_type: str,
        description: str = "",
        extra_params: Dict[str, Any] = None
    ):
        self.name = name
        self.impl_func = impl_func
        self.tool_type = tool_type
        self.description = description
        self.extra_params = extra_params or {}


def create_tool_wrapper(config: ToolConfig):
    """
    도구 래퍼 생성
    
    Args:
        config: 도구 구성
    
    Returns:
        래핑된 도구 함수
    """
    @econometric_tool(config.tool_type)
    async def tool_wrapper(ctx, **kwargs):
        """도구 래퍼 구현"""
        # 원본 구현 함수 호출
        return await config.impl_func(ctx, **kwargs)
    
    # 함수 속성 설정
    tool_wrapper.__name__ = config.name
    tool_wrapper.__doc__ = config.description
    
    return tool_wrapper


# 도구 유형 매개변수 정의
TOOL_TYPE_PARAMS = {
    "multi_var_dict": {
        "data": Annotated[
            Optional[Dict[str, List[float]]],
            Field(
                default=None, 
                description="""
다변수 데이터 사전

사용 방법:
- 사전 형식, 키는 변수명, 값은 숫자 목록
- 모든 변수는 동일한 길이여야 함
- 형식: {"변수1": [값1, 값2, ...], "변수2": [값1, 값2, ...]}

주의 사항:
- 변수명 중복 불가
- 데이터 길이 일치 필요
- 결측치 처리 지원
- 최소 1개 변수 필요

예시:
{"GDP": [100.5, 102.3, 104.1], "CPI": [2.1, 2.3, 2.2]}

적용 시나리오:
- 다변수 통계 분석
- 상관성 분석
- 주성분 분석"""
            )
        ]
    },
    "regression": {
        "y_data": Annotated[
            Optional[List[float]],
            Field(
                default=None, 
                description="""
종속 변수 데이터

사용 방법:
- 숫자 목록 형식
- x_data와 길이 일치해야 함
- 결측치 지원

주의 사항:
- NaN 값 포함 불가
- 길이가 x_data와 일치해야 함
- 최소 n+2개 관측치 필요(n은 독립 변수 개수)

예시:
[12.5, 13.2, 14.1, 13.8, 15.0]  # 종속 변수 데이터

적용 시나리오:
- OLS 회귀 분석
- 다중 회귀
- 예측 모델링"""
            )
        ],
        "x_data": Annotated[
            Optional[List[List[float]]],
            Field(
                default=None, 
                description="""
독립 변수 데이터 행렬

사용 방법:
- 2차원 목록 형식
- 각 행은 하나의 관측치, 각 열은 하나의 독립 변수
- 형식: [[x1_1, x2_1, ...], [x1_2, x2_2, ...], ...]

주의 사항:
- y_data 길이와 일치해야 함
- 여러 독립 변수 지원
- 상수항 포함 가능
- 최소 2개 관측치 필요

예시:
[[100, 50, 3],    # 관측치1: 독립 변수1=100, 독립 변수2=50, 독립 변수3=3
 [120, 48, 3],    # 관측치2: 독립 변수1=120, 독립 변수2=48, 독립 변수3=3
 [110, 52, 4]]    # 관측치3: 독립 변수1=110, 독립 변수2=52, 독립 변수3=4

적용 시나리오:
- OLS 회귀 분석
- 다중 회귀
- 특징 공학"""
            )
        ],
        "feature_names": Annotated[
            Optional[List[str]],
            Field(
                default=None, 
                description="""
독립 변수 이름 목록

사용 방법:
- 문자열 목록 형식
- x_data 열 수와 일치해야 함
- 결과 표시 및 해석에 사용

주의 사항:
["가격", "수입", "광고 지출", "계절 요인"]

적용 시나리오:
- 결과 해석
- 특징 중요성 분석
- 모델 보고서
- 제공하지 않으면 자동으로 x1, x2, x3...

예시:
- 결과 가독성 향상
- 모델 해석 용이
- 특징 선택"""
            )
        ]
    },
    "single_var": {
        "data": Annotated[
            Optional[List[float]],
            Field(
                default=None, 
                description="""
단일 변수 시간 시리즈 데이터

사용 방법:
- 숫자 목록 형식
- 시간 순서로 배열
- 결측치 처리 지원

주의 사항:
- 데이터 비어 있으면 안 됨
- 최소 5개 관측치 필요
- ARIMA 모델은 30+개 관측치 필요

예시:
[100.5, 102.3, 101.8, 103.5, 104.2, 103.8, 105.1]  # 시간 시리즈 데이터

적용 시나리오:
- 기술 통계
- ADF/KPSS 안정성 검정
- ACF/PACF 자기 상관 분석
- ARIMA/GARCH 모델링
- 시간 시리즈 예측"""
            )
        ]
    },
    "panel": {
        "y_data": Annotated[
            Optional[List[float]],
            Field(
                default=None, 
                description="""
패널 데이터 종속 변수

사용 방법:
- 숫자 목록 형식
- 데이터 길이 = 엔티티 수 × 시간 기간 수
- 엔티티-시간 순서로 배열

주의 사항:
1. [엔티티1-시간1, 엔티티1-시간2, ...]
2. [엔티티2-시간1, 엔티티2-시간2, ...]
3. entity_ids와 time_periods가 일치해야 함

예시 3엔티티2기간:
[1000, 1100, 800, 900, 1200, 1300]  # 종속 변수 데이터
엔티티1: 2020년=1000, 2021년=1100
엔티티2: 2020년=800,  2021년=900
엔티티3: 2020년=1200, 2021년=1300

적용 시나리오:
- 패널 회귀 분석
- 고정 효과 모델
- 무작위 효과 모델"""
            )
        ],
        "x_data": Annotated[
            Optional[List[List[float]]],
            Field(
                default=None, 
                description="""
패널 데이터 독립 변수 행렬

사용 방법:
- 2차원 목록 형식
- 데이터 길이 = 엔티티 수 × 시간 기간 수
- 각 행은 하나의 엔티티-시간 관측치
- y_data 길이와 일치해야 함

주의 사항:
- y_data 길이와 일치해야 함
- 여러 독립 변수 지원
- 상수항 포함 가능

예시 3엔티티2기간2독립 변수:
[[50, 100],   # 엔티티1-2020: 독립 변수1=50, 독립 변수2=100
 [52, 110],   # 엔티티1-2021: 독립 변수1=52, 독립 변수2=110
 [40, 80],    # 엔티티2-2020
 [42, 90],    # 엔티티2-2021
 [60, 150],   # 엔티티3-2020
 [62, 160]]   # 엔티티3-2021

적용 시나리오:
- 패널 회귀 분석
- 고정 효과 모델"""
            )
        ],
        "entity_ids": Annotated[
            Optional[List[str]],
            Field(
                default=None, 
                description="""
엔티티 식별자 목록

사용 방법:
- 문자열 목록 형식
- 각 관측치가 속한 엔티티 식별
- 길이가 y_data와 같아야 함

주의 사항:
- 제공해야 함
- y_data와 x_data와 일치해야 함
- 중복 식별자 지원

예시 3엔티티2기간:
["A", "A", "B", "B", "C", "C"]

["1", "1", "2", "2", "3", "3"]

적용 시나리오:
- 엔티티 효과 모델링
- 고정 효과 검정
- Hausman 검정

주의 사항:
- 식별자 제공 필요
- 문자열 및 숫자 식별자 지원"""
            )
        ],
        "time_periods": Annotated[
            Optional[List[str]],
            Field(
                default=None, 
                description="""
시간 기간 식별자 목록

사용 방법:
- 문자열 목록 형식
- 각 관측치가 속한 시간 기간 식별
- 길이가 y_data와 같아야 함

주의 사항:
- 제공해야 함
- y_data와 x_data와 일치해야 함
- 중복 식별자 지원

예시 3엔티티2기간:
["2020", "2021", "2020", "2021", "2020", "2021"]

["2020-01", "2020-02", "2020-01", "2020-02", "2020-01", "2020-02"]

적용 시나리오:
- 시간 효과 모델링
- 시간 고정 효과
- 패널 데이터 균형성 검정

주의 사항:
- 시간 식별자 제공 필요
- 다양한 시간 형식 지원"""
            )
        ],
        "feature_names": Annotated[
            Optional[List[str]],
            Field(
                default=None, 
                description="""regression 유형과 동일

사용 방법:
- 문자열 목록 형식
- x_data 열 수와 일치해야 함

주의 사항:
["가격", "수입", "광고 지출", "계절 요인"]

적용 시나리오:
- 결과 해석
- 특징 중요성 분석"""
            )
        ]
    },
    "time_series": {
        "data": Annotated[
            Optional[Dict[str, List[float]]],
            Field(
                default=None, 
                description="""
다변수 시간 시리즈 데이터

사용 방법:
- 사전 형식, 키는 변수명, 값은 시간 시리즈
- 모든 변수는 동일한 길이여야 함
- 여러 시간 시리즈 변수 지원

주의 사항:
- VAR/VECM을 위한 최소 2개 변수 필요
- single_var 유형과 다름
- 데이터는 시간 순서로 배열되어야 함
- VAR 모델은 40+개 관측치 필요
- VECM 모델은 60+개 관측치 필요

예시:
{
    "GDP": [100.5, 102.3, 104.1, 105.8],     # GDP 시간 시리즈
    "CPI": [2.1, 2.3, 2.2, 2.4],             # 소비자 가격 지수
    "실업률": [3.5, 3.6, 3.4, 3.7]             # 실업률
}

적용 시나리오:
- VAR 모델 분석
- VECM 공정성 분석
- 다변수 시간 시리즈 모델링
- Granger 인과성 검정
- 펄스 반응 분석

주의 사항:
- 변수 간 공정성 관계 존재 가능
- 충분한 시간 시리즈 길이 필요
- VAR 모델은 안정성 요구
- VECM 모델은 비안정 시퀀스 처리"""
            )
        ]
    }
}


def get_tool_params(tool_type: str, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    도구 매개변수 구성 가져오기
    
    Args:
        tool_type: 도구 유형
        extra_params: 추가 매개변수
    
    Returns:
        매개변수 구성 사전
    """
    params = {}
    
    # 기본 매개변수
    if tool_type in TOOL_TYPE_PARAMS:
        params.update(TOOL_TYPE_PARAMS[tool_type])
    
    # 파일 입력 매개변수
    params.update(FILE_INPUT_PARAMS)
    
    # 추가 매개변수
    if extra_params:
        params.update(extra_params)
    
    return params