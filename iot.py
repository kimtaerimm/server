import cv2
import requests
import time

# 서버 주소 (태림님의 컴퓨터 IP 주소)
SERVER_URL = "http://172.30.1.43:8000/iot/predict"

# 1. 카메라 연결
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # 2. 이미지 임시 저장 또는 메모리 버퍼로 변환
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        # 3. 서버로 전송!
        # 3. 서버로 전송!
        try:
            response = requests.post(SERVER_URL, files=files)
            data = response.json()
            
            # 4. 서버가 분석한 결과 처리
            results = data.get("results", []) # 결과 리스트 가져오기 (없으면 빈 리스트)

            if results: # 감지된 게 하나라도 있을 때
                severity = results[0].get("class") # 첫 번째 감지된 물체의 클래스
                print(f"📡 분석 결과 감지: {severity}")
                
                if severity == "High":
                    print("🚨 심각한 파손 발견! 경고등 켜기!")
            else:
                print("🔍 감지된 물체나 파손이 없습니다.")

        except Exception as e:
            print(f"서버 연결 실패: {e}")
            
    time.sleep(5) # 5초마다 한 번씩 점검