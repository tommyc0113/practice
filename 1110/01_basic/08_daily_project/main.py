
"""
Knowledge API
실행 명령:
python -m uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from storage import LOCK, read_data, write_data


# --------------------------------------------------
# 1. 데이터 입력 모델
# --------------------------------------------------

class KnowledgeWrite(BaseModel):
    # 제목: 1~80자
    title: str = Field(min_length=1, max_length=80)

    # 본문: 1~2000자
    content: str = Field(min_length=1, max_length=2000)

    # 분류: 1~30자
    category: str = Field(min_length=1, max_length=30)


# 응답 모델: 입력 모델에 ID 추가
class Knowledge(KnowledgeWrite):
    id: int


# --------------------------------------------------
# 2. FastAPI 애플리케이션 생성
# --------------------------------------------------

app = FastAPI(
    title="Knowledge API",
    description="AI 플랫폼의 원본 지식 관리 API",
    version="1.0.0"
)


# --------------------------------------------------
# 3. 서버 상태 확인
# --------------------------------------------------

@app.get("/health", tags=["공통"])
def health():
    return {
        "status": "ok",
        "message": "Knowledge API 서버 정상 실행"
    }


# --------------------------------------------------
# 4. CREATE - 지식 데이터 생성
# --------------------------------------------------

@app.post(
    "/api/knowledge",
    status_code=201,
    response_model=Knowledge,
    tags=["Knowledge CRUD"]
)
def create_knowledge(payload: KnowledgeWrite):

    with LOCK:
        # 기존 데이터 읽기
        data = read_data()

        # 요청 데이터를 딕셔너리로 변환
        entry = payload.model_dump()

        # 새로운 ID 발급
        entry["id"] = data["next_id"]

        # 데이터 목록에 추가
        data["entries"].append(entry)

        # 다음 ID 증가
        data["next_id"] += 1

        # JSON 파일에 저장
        write_data(data)

    return entry


# --------------------------------------------------
# 5. READ - 전체 지식 데이터 조회
# --------------------------------------------------

@app.get(
    "/api/knowledge",
    tags=["Knowledge CRUD"]
)
def list_knowledge():

    with LOCK:
        data = read_data()

    # 내부 관리용 next_id는 제외하고 entries만 반환
    return data["entries"]


# --------------------------------------------------
# 6. READ - 특정 지식 데이터 조회
# --------------------------------------------------

@app.get(
    "/api/knowledge/{knowledge_id}",
    tags=["Knowledge CRUD"]
)
def get_knowledge(knowledge_id: int):

    with LOCK:
        data = read_data()

    # ID가 일치하는 데이터 검색
    for entry in data["entries"]:

        if entry["id"] == knowledge_id:
            return entry

    # 데이터가 없으면 404 오류
    raise HTTPException(
        status_code=404,
        detail="Knowledge not found"
    )


# --------------------------------------------------
# 7. UPDATE - 기존 지식 데이터 수정
# --------------------------------------------------

@app.put(
    "/api/knowledge/{knowledge_id}",
    response_model=Knowledge,
    tags=["Knowledge CRUD"]
)
def update_knowledge(
    knowledge_id: int,
    payload: KnowledgeWrite
):

    with LOCK:
        data = read_data()

        # 데이터 위치와 항목을 함께 조회
        for i, entry in enumerate(data["entries"]):

            if entry["id"] == knowledge_id:

                # 새 요청 데이터 생성
                updated = payload.model_dump()

                # 기존 ID 유지
                updated["id"] = knowledge_id

                # 기존 데이터 교체
                data["entries"][i] = updated

                # 변경 데이터 저장
                write_data(data)

                return updated

    # 해당 ID가 없으면 404
    raise HTTPException(
        status_code=404,
        detail="Knowledge not found"
    )


# --------------------------------------------------
# 8. DELETE - 지식 데이터 삭제
# --------------------------------------------------

@app.delete(
    "/api/knowledge/{knowledge_id}",
    status_code=204,
    tags=["Knowledge CRUD"]
)
def delete_knowledge(knowledge_id: int):

    with LOCK:
        data = read_data()

        for i, entry in enumerate(data["entries"]):

            if entry["id"] == knowledge_id:

                # 해당 위치의 데이터 삭제
                data["entries"].pop(i)

                # 삭제 결과 저장
                write_data(data)

                # 빈 204 응답 반환
                return Response(status_code=204)

    # 삭제할 데이터가 없으면 404
    raise HTTPException(
        status_code=404,
        detail="Knowledge not found"
    )
