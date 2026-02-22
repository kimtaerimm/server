from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageDraw, ImageStat
import io
import time
import random
import os
from datetime import datetime
import base64

app = FastAPI()

if not os.path.exists("server_results"):
    os.makedirs("server_results")

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: sans-serif; background: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .card { background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); text-align: center; width: 90%; max-width: 400px; }
            .upload-zone { border: 2px dashed #007bff; padding: 20px; border-radius: 15px; cursor: pointer; margin-bottom: 20px; }
            .btn { background: #007bff; color: white; border: none; padding: 12px 20px; border-radius: 10px; font-weight: bold; cursor: pointer; width: 100%; }
            #preview { width: 100%; display: none; margin-top: 15px; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🚗 가상 AI 진단기</h2>
            <p>사진 데이터를 분석해 결과를 도출합니다.</p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" id="fileInput" style="display:none" onchange="showPreview(event)">
                <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                    <span>📷 사진 업로드</span>
                    <img id="preview">
                </div>
                <button type="submit" class="btn">분석 시작 (데이터 기반)</button>
            </form>
        </div>
        <script>
            function showPreview(event) {
                const reader = new FileReader();
                reader.onload = function() {
                    const output = document.getElementById('preview');
                    output.src = reader.result;
                    output.style.display = 'block';
                };
                reader.readAsDataURL(event.target.files[0]);
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    # 1. 이미지 읽기
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # 2. [가상 모델 로직] 이미지의 통계 데이터 추출
    # 이미지 전체의 평균 밝기를 계산 (0: 검정, 255: 흰색)
    stat = ImageStat.Stat(image)
    avg_brightness = sum(stat.mean) / 3 
    
    # 3. 밝기에 따른 가상 분석 (어두운 영역에 박스 치기 시뮬레이션)
    # 실제 모델이 들어갈 자리입니다.
    box_size = 150
    left = random.randint(0, width - box_size)
    top = random.randint(0, height - box_size)
    
    # 평균 밝기가 낮을수록(어두울수록) 심각도를 높게 책정하는 척 함
    if avg_brightness < 100:
        severity = "Severe (중형 파손)"
        color = "red"
    else:
        severity = "Minor (경미한 스크래치)"
        color = "green"

    draw.rectangle([left, top, left+box_size, top+box_size], outline=color, width=8)

    # 4. 결과 이미지 저장 및 Base64 변환
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
    <body style="text-align:center; font-family:sans-serif; padding:20px;">
        <h2>분석 완료</h2>
        <img src="data:image/jpeg;base64,{img_str}" style="max-width:100%; border-radius:15px;">
        <div style="margin-top:20px; padding:20px; border-radius:15px; background:#f8f9fa;">
            <p>데이터 분석 기반 예측</p>
            <h1 style="color:{color};">{severity}</h1>
            <p>평균 이미지 밀도: {avg_brightness:.2f}</p>
        </div>
        <br><a href="/" style="text-decoration:none; color:#007bff; font-weight:bold;">🔄 다시 시도</a>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)