import os
import configparser
import torch
import cv2
import collections
from PIL import Image
import torchvision.transforms as transforms
from Age_Estimation.model2 import MultipleOutputCNN  # 사용자의 모델 정의 파일 임포트
import time

# 장치 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 나이 범주 매핑 함수 정의
def map_age_to_category(age):
    if age == 1:
        return 'u20'
    elif age == 2:
        return '20s'
    elif age == 3:
        return '30s'
    elif age == 4:
        return '40s'
    else:
        return 'over50'

# 얼굴을 감지하고 박스를 그리는 함수
def highlight_face(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]

    # DNN 입력 블롭 생성 (이미지를 DNN에서 처리 가능한 포맷으로 변환)
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()

    face_boxes = []
    max_box = None
    max_area = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)

            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame_width - 1, x2 + padding)
            y2 = min(frame_height - 1, y2 + padding)

            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                max_box = [x1, y1, x2, y2]

    if max_box:
        face_boxes = [max_box]
    return face_boxes


# 프레임에서 얼굴을 감지하고 나이를 예측하는 함수
def process_frame(face_net, age_net, frame, transform, padding=20):
    face_boxes = highlight_face(face_net, frame)
    data = []

    for face_box in face_boxes:
        face = frame[max(0, face_box[1] - padding): min(face_box[3] + padding, frame.shape[0] - 1),
                    max(0, face_box[0] - padding): min(face_box[2] + padding, frame.shape[1] - 1)]

        # 얼굴 이미지를 PIL 형식으로 변환 후 전처리
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # 나이 예측
        with torch.no_grad():
            output = age_net(face_tensor)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            predict_age = torch.sum(output, dim=1)[:, 0] + 1
            age_category = map_age_to_category(predict_age.item())

        data.append(age_category)

    return data

# 얼굴 검출 DNN 모델 파일 경로 설정
face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"

# 얼굴 검출 DNN 모델 로드
face_net = cv2.dnn.readNet(face_model, face_proto)

# 나이 예측 모델 로드 (사용자 학습 모델)
age_net = MultipleOutputCNN().to(device)
age_net.load_state_dict(torch.load('./0.865_5train_best.pth', map_location=device))
age_net.eval()

# 이미지 전처리 설정 (사용자 학습 모델의 입력에 맞춤)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((60, 60)),
])

# AI.ini 파일을 경로에서 불러와 읽습니다.
OUTPUT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AI.ini')
output = configparser.ConfigParser()
ret = output.read(OUTPUT_FILE_PATH)

# 만약 실패하면 프로그램을 종료합니다.
if not ret:
    print(f"{OUTPUT_FILE_PATH} 파일이 존재하지 않거나, 읽을 수 없습니다.")
    exit(0)

# AI모델 로딩이 완료되면, AI.ini의 loaded 항목을 1로 변경합니다.
output.set('MODEL', 'loaded', '1')
with open(OUTPUT_FILE_PATH, 'w') as ini:
    output.write(ini)

# 메인 루프
while True:
    # 루프 내에서 실시간으로 AI.ini를 확인합니다.
    ret = output.read(OUTPUT_FILE_PATH)
    if not ret:
        break

    # 명령어를 표준 입력으로 받아들입니다.
    user_input = input()

    # 입력이 'face'라면, 얼굴 인식을 수행합니다.
    if user_input == 'face' and output['FACE']['recognized'] == '0':
        # 비디오 캡처 객체 생성 (웹캠 사용)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            exit()

        print("캠이 작동 중입니다.")  # 캠 시작 문구 출력

        age_predictions = collections.deque(maxlen=10)
        stable_start_time = None   # 숫자가 안정적으로 유지되기 시작한 시간
        most_common_age = "Unknown"

        # 얼굴 인식 및 나이 예측 수행
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            # 얼굴 인식 및 나이 예측 수행
            predicted_ages = process_frame(face_net, age_net, frame, transform)

            # 예측된 나이 범주를 큐에 추가 (여러 얼굴이 있는 경우, 각 얼굴의 예측값을 추가)
            age_predictions.extend(predicted_ages)

            # 얼굴이 인식되었다면
            if predicted_ages:
                most_common_age = collections.Counter(age_predictions).most_common(1)[0][0] if age_predictions else "Unknown"

                # age_predictions를 리스트로 변환 (크기 확인)
                most_common_age = (
                    collections.Counter(list(age_predictions)).most_common(1)[0][0]
                    if len(age_predictions) > 0  # 크기를 확인하려면 len() 사용
                    else "Unknown"
                )

                if stable_start_time is None:
                    stable_start_time = time.time()
                elif time.time() - stable_start_time >= 3:  # 3초 동안 동일한 예측값이 유지되는지 확인
                    print(f"나이 {most_common_age}가 3초 동안 유지되었습니다.")
                    break
            else:
                stable_start_time = None


        # 리소스 해제
        cap.release()

        # 캠 종료 후 ini 파일 업데이트
        output.set('FACE', 'recognized', '1')
        output.set('FACE', 'age', most_common_age)
        output.set('FACE', 'sex', 'Unknown')  # 성별 예측은 구현되지 않았으므로 기본값 설정

        # .ini 파일 업데이트
        with open(OUTPUT_FILE_PATH, 'w') as ini:
            output.write(ini)
        print("인식된 숫자가 AI.ini에 기록되었습니다.")

    # 입력이 'quit'라면, 루프를 빠져 나갑니다.
    if user_input == 'quit':
        break

print("프로그램이 종료되었습니다.")