"""실습 03. 정상 라벨의 면적·비율·클래스 수와 그룹 분리를 확인합니다."""
from pathlib import Path
import pandas as pd
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True)
manifest=pd.read_csv(BASE/"data/manifest.csv")
rows=[]
for image in manifest.itertuples():
    for line in (BASE/"data"/image.label).read_text(encoding="utf-8").splitlines():
        cls,cx,cy,w,h=map(float,line.split())
        # >>> A03
        area = w * h # 정규화된 면적 계산
        aspect_ratio = (w * image.width) / (h * image.height) # 원본 픽셀 기준 가로세로비 계산

        # 요구: 정규화 면적 w*h와 원본 pixel 기준 가로세로비를 기록하세요.
        # 해설: 비정방형 이미지에서는 w/h만 쓰면 원본 비율이 아닙니다.
        # 객체별 Bounding Box의 크기와 가로세로 비율을 계산하여 분석 결과에 추가
        #
        # image.image_id : 이미지 식별자(ID)
        # image.split    : 데이터 구분 (train / valid / test)
        # class_id       : 객체 클래스 번호 (예: 0=square, 1=circle, 2=triangle)
        #
        # area_ratio = w * h
        #   → 이미지 전체 면적 대비 Bounding Box가 차지하는 비율
        #   → w, h는 YOLO 형식의 정규화된 너비와 높이(0~1)
        #   → 예: w=0.2, h=0.3이면 면적 비율은 0.06(6%)
        #
        # aspect_ratio = (w * image.width) / (h * image.height)
        #   → Bounding Box의 실제 픽셀 너비 / 실제 픽셀 높이
        #   → 정규화된 w, h를 각각 이미지 너비·높이로 환산한 후 계산
        #   → 1.0 : 가로와 세로 길이가 동일한 박스
        #   → 1.0 초과 : 가로가 더 긴 박스
        #   → 1.0 미만 : 세로가 더 긴 박스
        #
        # rows.append(): 객체별 분석 결과를 딕셔너리 형태로 리스트에 저장
        # 주의: h는 0보다 커야 하며, 박스가 이미지 내부에 있다는 전제에서 해석
        rows.append({
            "split": image.split,
            "class_id": int(cls),
            "area": area,
            "aspect_ratio": aspect_ratio
        })
# <<< A03
r=pd.DataFrame(rows);r.to_csv(out/"label_audit.csv",index=False)
print(r.groupby(["split","class_id"]).size().unstack(fill_value=0).to_string())
# split별 클래스 개수확인
"""
groupby(): split과 클래스별로 그룹화
size(): 그룹별 데이터 개수 계산
unstack(): 클래스를 열 방향으로 펼침
fill_value=0: 데이터가 없는 클래스는 0으로 표시
"""
print("split 간 그룹 중복:", int((manifest.groupby("group_id").split.nunique()>1).sum()))
print("정확히 같은 파일:", int(manifest.sha256.duplicated().sum()))
print("빈 라벨은 정상 negative 이미지이며 누락 라벨과 다릅니다.")
