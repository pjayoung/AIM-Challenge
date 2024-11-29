# 얼굴 인식 함수
def do_face(image_file_path: str) -> str:
    import torch
    import cv2
    from PIL import Image
    import torchvision.transforms as transforms
    from Age_Estimation.model2 import MultipleOutputCNN

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

    # 장치 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 얼굴 검출 모델 파일 경로 설정
    face_proto = "opencv_face_detector.pbtxt"
    face_model = "opencv_face_detector_uint8.pb"

    # 얼굴 검출 모델 로드
    face_net = cv2.dnn.readNet(face_model, face_proto)

    # 나이 예측 모델 로드
    age_net = MultipleOutputCNN().to(device)
    age_net.load_state_dict(torch.load('./0.865_5train_best.pth', map_location=device))
    age_net.eval()

    # 이미지 전처리 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((60, 60)),
    ])

    # 이미지 읽기
    frame = cv2.imread(image_file_path)
    if frame is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_file_path}")

    # 얼굴 감지
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]

    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

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

    if not face_boxes:
        return f"얼굴을 감지하지 못했습니다: {image_file_path}"
        

    for face_box in face_boxes:
        x1, y1, x2, y2 = face_box
        if x2 <= x1 or y2 <= y1:  # 잘못된 바운딩 박스 필터링
            return f"잘못된 바운딩 박스: {face_box}"
            
        face = frame[y1:y2, x1:x2]
        if face.size == 0:  # 빈 이미지를 필터링
            return f"빈 얼굴 이미지: {image_file_path}"
                
    
    # 나이 예측 수행
    age_categories = []
    for face_box in face_boxes:
        face = frame[max(0, face_box[1]): min(face_box[3], frame.shape[0] - 1),
                     max(0, face_box[0]): min(face_box[2], frame.shape[1] - 1)]

        # 얼굴 이미지를 PIL 형식으로 변환 후 전처리
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = age_net(face_tensor)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            predict_age = torch.sum(output, dim=1)[:, 0] + 1
            age_category = map_age_to_category(predict_age.item())
            age_categories.append(age_category)

    if age_categories:
        return age_categories[0]  # 첫 번째 얼굴의 나이 범주를 반환
    else:
        return "No face detected"

# 사용 예시
result = do_face("./12575A57.jpg")
print(result)  # 예: '20s'