"""실습 04: 이미지+질문을 구성하고 모의 응답을 읽습니다. 실제 VLM을 호출하지 않습니다."""
from pathlib import Path
import json

base = Path(__file__).resolve().parent
lesson = json.loads((base / 'lesson.json').read_text(encoding='utf-8'))
doc = lesson['documents']['doc01']
rows = []
for answer in doc['mock_vlm']['responses']:
    question = lesson['questions'][answer['field']]
    # 해설 B05
        # VLM(Vision-Language Model)에 전달할 메시지 구성
    #
    # messages : AI 모델에 전달할 대화 메시지 목록
    # role     : 'user'는 사용자가 전달한 입력 메시지를 의미
    # content  : 하나의 메시지에 이미지와 텍스트를 함께 포함
    #
    # ① {'type': 'image', 'path': ...}
    #    → 분석할 문서 이미지(doc01.png)의 파일 경로 전달
    #    → 모델이 이미지에 포함된 글자, 표, 문서 구조 등을 분석하도록 함
    #
    # ② {'type': 'text', 'text': question}
    #    → 이미지에 대해 사용자가 작성한 질문을 함께 전달
    #    → 예: "이 문서의 날짜와 총금액을 추출해줘."
    #
    # 목적:
    # 이미지와 질문을 하나의 메시지로 묶어 VLM이
    # 시각 정보와 자연어 질문을 함께 처리하도록 구성
    #
    # 주의: 실제 이미지 입력 형식은 사용하는 VLM 라이브러리에 따라 다름

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'path': str(base / 'doc01.png')},
                {'type': 'text', 'text': question}
            ]
        }
    ]
    rows.append({'backend': 'mock', 'messages': messages, 'response': answer, 'model_id': None, 'inference_ms': None})
    print(question, '->', answer['answer'])
(base / 'p04_mock_answers.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print('MOCK: 교사가 작성한 응답 예시입니다. 실제 모델 출력이 아닙니다.')