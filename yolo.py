from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import base64

app = FastAPI()

# 1. 진짜 탐지용 AI 모델 로드 (YOLOv8)
# 처음 실행 시 모델 파일(6MB)을 자동으로 다운로드합니다.
model = YOLO("yolov8n.pt") 

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # 2. 진짜 AI 추론 (사진에서 물체와 위치를 찾아냄)
    results = model(image) 
    
    # 3. AI가 찾은 결과가 그려진 이미지를 가져옴
    res_plotted = results[0].plot() 
    res_image = Image.fromarray(res_plotted[:, :, ::-1]) # RGB 변환

    # 4. 화면 전송용 변환
    buffered = io.BytesIO()
    res_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"""
    <body style="text-align:center; background:#000; color:#fff;">
        <h1>🔥 진짜 AI 객체 탐지 모드</h1>
        <img src="data:image/jpeg;base64,{img_str}" style="max-width:80%;">
        <p>AI가 스스로 위치를 찾아 박스를 그렸습니다.</p>
        <a href="/" style="color:yellow;">다시 하기</a>
    </body>
    """