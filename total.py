from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import base64
import time

app = FastAPI()

print("AI 모델을 불러오는 중입니다...")
model = YOLO("yolov8n.pt") # [수정필요] AI 모델 로드 (서버 켜질 때 한 번만 준비)
print("준비 완료")

@app.get("/", response_class=HTMLResponse)
async def main():
    
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IoT Car Damage AI</title>
        <style>
            body {
                font-family: 'Noto Sans KR', sans-serif;
                background: linear-gradient(135deg, #eef2f7, #e8f1ff);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                color: #263238;
            }

            .container {
                background: rgba(255, 255, 255, 0.92);
                padding: 2rem;
                border-radius: 16px;
                box-shadow: 0 14px 35px rgba(0, 0, 0, 0.10);
                text-align: center;
                width: 350px;
                border: 1px solid rgba(226, 232, 240, 0.9);
                backdrop-filter: blur(6px);
            }

            h2 {
                color: #1f2d3d;
                margin: 0 0 14px;
                font-size: 20px;
                letter-spacing: -0.2px;
            }

            p {
                margin: 0 0 18px;
                color: #546e7a;
                font-size: 14px;
                line-height: 1.5;
            }

            input[type=file] {
                margin: 0 0 16px;
                width: 100%;
                font-size: 13px;
                color: #455a64;
                padding: 10px;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                background: #f8fafc;
                box-sizing: border-box;
            }

            input[type=file]::file-selector-button {
                border: none;
                padding: 8px 12px;
                border-radius: 8px;
                background: #e3f2fd;
                color: #1565c0;
                cursor: pointer;
                margin-right: 10px;
                transition: 0.2s;
                font-weight: 600;
            }

            input[type=file]::file-selector-button:hover {
                background: #d7ecff;
            }

            .btn {
                background: linear-gradient(135deg, #29b6f6, #0288d1);
                color: white;
                padding: 14px 15px;
                width: 100%;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                cursor: pointer;
                transition: transform 0.08s ease, filter 0.2s ease;
                font-weight: 700;
                letter-spacing: -0.2px;
            }

            .btn:hover {
                filter: brightness(1.05);
            }

            .btn:active {
                transform: translateY(1px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚗 IoT 차량 진단 서버</h2>
            <p>웹에서 테스트하거나<br>IoT 기기에서 데이터를 전송하세요.</p>
            <form action="/web/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <button type="submit" class="btn">분석 시작 (웹 모드)</button>
            </form>
        </div>
    </body>
    </html>
    """

# ---------------------------------------------------------
# [모드 2] 사람용: 웹 분석 결과 처리 (HTML 반환)
# ---------------------------------------------------------
@app.post("/web/predict", response_class=HTMLResponse)
async def web_predict(file: UploadFile = File(...)):
    start = time.time()
    
    # 이미지 읽기
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model(image) #AI모델을 통한 추론
    
    # 결과 이미지 생성
    res_plotted = results[0].plot()
    res_image = Image.fromarray(res_plotted[..., ::-1])
    
    # 웹 표시용 Base64 변환
    buffered = io.BytesIO()
    res_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # 감지된 객체 목록 텍스트 만들기
    detected_objects = []
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        detected_objects.append(f"{cls_name} ({conf*100:.1f}%)")
    
    obj_str = ", ".join(detected_objects) if detected_objects else "감지된 객체 없음"

    return f"""
    <body style="text-align:center; font-family:sans-serif; background:#263238; color:white; padding-top:40px;">
        <div style="background:#37474f; display:inline-block; padding:30px; border-radius:20px;">
            <h2 style="color:#29b6f6;">📡 분석 완료</h2>
            <img src="data:image/jpeg;base64,{img_str}" style="max-width:100%; border-radius:10px; border:3px solid #546e7a;">
            <div style="margin-top:20px; text-align:left; background:#455a64; padding:15px; border-radius:10px;">
                <p>⏱ 소요 시간: {time.time()-start:.4f}초</p>
                <p>🔍 감지 결과: <strong>{obj_str}</strong></p>
            </div>
            <br>
            <a href="/" style="color:#b0bec5; text-decoration:none;">다시 테스트하기</a>
        </div>
    </body>
    """

# ---------------------------------------------------------
# [모드 3] IoT 기계용: 데이터 통신 전용 (JSON 반환)
# ---------------------------------------------------------
@app.post("/iot/predict", response_class=JSONResponse)
async def iot_predict(file: UploadFile = File(...)):
    # IoT 기기가 보낸 이미지 읽기 
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    results = model(image) #AI모델 통해서 이미지에 대한 추론 수행
    
    detections = [] # 감지된 정보만 깔끔하게 정리해서 리스트로 만듦
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 2),
            "box": box.xyxy.tolist()[0]
        })
    
    response_data = { # 기계에게 보내줄 최종 데이터 (JSON)
        "status": "success",
        "count": len(detections),
        "results": detections,
        "message": "IoT device data processed successfully."
    }
    
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # 0.0.0.0으로 열어야 같은 와이파이의 IoT 기기가 접속 가능