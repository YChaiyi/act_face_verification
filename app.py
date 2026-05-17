from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
from PIL import Image
import numpy as np
import os
import dlib

# 检测是否有可用的GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# 初始化MTCNN和InceptionResnetV1
mtcnn = MTCNN()
# 对于在 CASIA-Webface 上预训练的模型
resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

# 加载大头照并提取特征向量
def load_and_extract_features(folder_path):
    face_database = {}
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if os.path.isdir(person_folder):
            embeddings = []
            for filename in os.listdir(person_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(person_folder, filename)
                    img = Image.open(img_path).convert('RGB')
                    # 使用MTCNN进行人脸检测和对齐
                    face_aligned = mtcnn(img)
                    if face_aligned is not None:
                        # 使用InceptionResnetV1提取特征向量
                        face_embedding = resnet(face_aligned.unsqueeze(0))
                        embeddings.append(face_embedding.detach().numpy())
            # 计算平均特征向量
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                face_database[person_name] = avg_embedding
    return face_database

# 比较两个特征向量之间的距离
def compare_faces(enc1, enc2):
    return np.linalg.norm(enc1 - enc2)

# 识别人脸
def recognize_face(face, database, threshold=1.0):
    try:
        # 检查传入的人脸图像是否为空
        if face is None or face.size == 0:
            return None

        # 转换图像颜色从BGR到RGB并调整大小
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        # 使用MTCNN进行人脸检测和对齐
        face_aligned = mtcnn(face_pil)
        if face_aligned is not None:
            # 使用InceptionResnetV1提取特征向量
            face_embedding = resnet(face_aligned.unsqueeze(0)).detach().numpy()

            # 初始化最小距离和识别标签
            min_dist = threshold
            identity = "unknown"  # 修改默认标签为unknown

            # 遍历数据库比较特征向量
            for name, db_enc in database.items():
                dist = compare_faces(face_embedding, db_enc)
                if dist < min_dist and dist <= 0.65:
                    min_dist = dist
                    identity = name

            return identity
    except Exception as e:
        return None


# 加载人脸库
folder_path = './person/'  # 设置为大头照所在文件夹的路径
face_database = load_and_extract_features(folder_path)

def mouth_aspect_ratio(mouth_points):
    """
    计算嘴巴的纵横比(Mouth Aspect Ratio, MAR)
    
    参数:
    - mouth_points: np.array, 包含嘴部关键点的坐标，形状为(20, 2)。

    返回:
    - mar: float, 嘴巴的纵横比。
    """
    # 计算嘴巴上部与下部之间三个垂直距离的平均值
    A = np.linalg.norm(mouth_points[13] - mouth_points[19])  # 上下唇之间的距离
    B = np.linalg.norm(mouth_points[14] - mouth_points[18])
    C = np.linalg.norm(mouth_points[15] - mouth_points[17])
    
    # 计算嘴巴左右两边之间的距离
    D = np.linalg.norm(mouth_points[12] - mouth_points[16])  # 嘴角之间的距离
    
    # 计算嘴巴的纵横比
    mar = (A + B + C) / (2.0 * D)  # 嘴巴纵横比的计算公式
    
    return mar

def is_real_person(face, detector, predictor):
    """
    判断给定的人脸图像中的人是否张嘴
    
    参数:
    - face: np.array, 人脸图像。
    - detector: dlib的人脸检测器。
    - predictor: dlib的人脸关键点预测器。
    
    返回:
    - bool, 如果检测到张嘴则返回True，否则返回False。
    """
    # 将图像转换为灰度图，以减少计算复杂度
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # 使用detector检测图像中的人脸
    rects = detector(gray, 0)
    
    # 遍历检测到的每个人脸
    for rect in rects:
        # 使用predictor获取人脸关键点
        shape = predictor(gray, rect)
        
        # 将dlib的形状对象转换为numpy数组，方便处理
        shape = np.matrix([[p.x, p.y] for p in shape.parts()])
        
        # 提取嘴部的关键点（一共68个关键点中的第49到第68点）
        mouth = shape[48:68]
        
        # 计算嘴巴的纵横比
        mar = mouth_aspect_ratio(mouth)
        
        # 设定张嘴的阈值，用于判断是否张嘴
        MAR_THRESHOLD = 0.6  # 嘴巴纵横比的阈值，根据需要调整
        
        # 如果嘴巴的纵横比大于阈值，则认为是张嘴的
        if mar > MAR_THRESHOLD:
            return True
    
    # 如果没有检测到张嘴的人脸，返回False
    return False


# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低帧的宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 降低帧的高度

frame_skip = 1  # 每处理一帧，跳过几帧
frame_count = 0

# 在主循环之前初始化dlib的面部检测器和面部标志检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

liveness_passed = False  # 活体检测通过的标志

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if not liveness_passed:
                # 在活体检测未通过之前，显示提示让用户张嘴
                cv2.putText(frame, "Please open your mouth", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            face = frame[y1:y2, x1:x2]

            if not liveness_passed:
                # 如果还没有通过活体检测，检查是否张嘴
                liveness_passed = is_real_person(face, detector, predictor)

            if liveness_passed:
                # 如果通过了活体检测，进行人脸识别并显示绿色框和名称
                identity = recognize_face(face, face_database)
                color = (0, 0, 255) if identity == "unknown" else (0, 255, 0)  # 红色用于known_person，绿色用于其他
                cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        liveness_passed = False  # 如果没有检测到人脸，重置活体检测状态

    cv2.imshow('人脸识别', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()