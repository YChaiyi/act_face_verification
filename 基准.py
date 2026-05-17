from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
from PIL import Image
import numpy as np
import os

# 检测是否有可用的GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# 初始化MTCNN和InceptionResnetV1
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低帧的宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 降低帧的高度

frame_skip = 1  # 每处理一帧，跳过几帧
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # 跳过部分帧以减少处理量

    # 使用MTCNN检测人脸
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # 确保坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue  # 如果计算出的坐标不合理，则跳过当前框
            face = frame[y1:y2, x1:x2]

            # 识别人脸
            identity = recognize_face(face, face_database)
            if identity is not None:
                color = (0, 0, 255) if identity == "unknown" else (0, 255, 0)  # 红色用于known_person，绿色用于其他
                cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 显示结果
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头
cap.release()
cv2.destroyAllWindows()