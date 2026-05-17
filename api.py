from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import io
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
import dlib
import logging
import uuid
import torch

app = Flask(__name__)
CORS(app)

# 检测是否有可用的GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# 初始化MTCNN和InceptionResnetV1
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 图片保存目录
UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 加载人脸检测器
detector = dlib.get_frontal_face_detector()

# 预定义的颜色和它们的RGB值
COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255)
}

# 颜色相似度的阈值
COLOR_THRESHOLD = 187

# 存储颜色检测结果的字典
color_results = {}

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

def compare_faces(enc1, enc2):
    return np.linalg.norm(enc1 - enc2)

def recognize_face(face, database, threshold=1.0):
    try:
        if face is None or face.size == 0:
            return None

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_aligned = mtcnn(face_pil)
        if face_aligned is not None:
            face_embedding = resnet(face_aligned.unsqueeze(0)).detach().numpy()

            min_dist = threshold
            identity = "unknown"

            for name, db_enc in database.items():
                dist = compare_faces(face_embedding, db_enc)
                if dist < min_dist and dist <= 0.65:
                    min_dist = dist
                    identity = name

            return identity
    except Exception as e:
        return None

def get_face_average_color(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        face = faces[0]
        face_image = image[face.top():face.bottom(), face.left():face.right()]
        avg_color_per_row = np.average(face_image, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        return avg_color.astype(int)
    else:
        return None

def is_color_similar(color1, color2, threshold):
    distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    return bool(distance <= threshold)

def mouth_aspect_ratio(mouth_points):
    A = np.linalg.norm(mouth_points[13] - mouth_points[19])
    B = np.linalg.norm(mouth_points[14] - mouth_points[18])
    C = np.linalg.norm(mouth_points[15] - mouth_points[17])
    D = np.linalg.norm(mouth_points[12] - mouth_points[16])
    mar = (A + B + C) / (2.0 * D)
    return mar

def is_real_person(face, detector, predictor):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.matrix([[p.x, p.y] for p in shape.parts()])
        mouth = shape[48:68]
        mar = mouth_aspect_ratio(mouth)
        MAR_THRESHOLD = 0.65
        if mar > MAR_THRESHOLD:
            return True
    return False

@app.route('/check_mouth_open', methods=['POST'])
def check_mouth_open():
    file = request.files['file']
    if not file:
        return jsonify({'message': 'No file received'}), 400

    # 将文件读取为图像
    image = Image.open(file.stream)
    image_np = np.array(image)

    # 检测是否张嘴
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 确保这个路径正确
    is_open = is_real_person(image_np, detector, predictor)

    return jsonify({'mouth_open': is_open})

@app.route('/check_color', methods=['POST'])
def check_color():
    color_name = request.form['color']
    file = request.files['image']
    if not file:
        return jsonify({'message': 'No file received'}), 400

    image = Image.open(file.stream)
    image_np = np.array(image.convert('RGB'))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 获取人脸区域的平均颜色
    face_avg_color = get_face_average_color(image_np)
    if face_avg_color is None:
        return jsonify({'error': 'No face detected'}), 400

    # 检查颜色是否相近
    is_real = is_color_similar(COLORS[color_name], face_avg_color, COLOR_THRESHOLD)

    return jsonify({'is_real': is_real})

@app.route('/find_name', methods=['POST'])
def find_name():
    file = request.files['file']
    if not file:
        return jsonify({'message': 'No file received'}), 400

    # 将上传的图像转换为OpenCV格式
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    image = cv2.imdecode(data, color_image_flag)

    # 识别人脸
    identity = recognize_face(image, face_database)

    return jsonify({'identity': identity})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    file = request.files['file']
    name = request.form.get('name', '')
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and name:
        # 读取图片
        img = Image.open(file.stream)
        
        # 使用MTCNN检测人脸
        boxes, _ = mtcnn.detect(img)
        
        # 检查是否只有一个人脸
        if boxes is not None and len(boxes) == 1:
            # 检查目标文件夹是否存在
            target_dir = os.path.join('/Users/liuhaiyi/Library/CloudStorage/OneDrive-个人/haiyi文件/学校/个人/英才计划/2024寒假线下活动/工坊/person', name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # 生成唯一文件名
            filename = f"{uuid.uuid4()}.jpg"
            file_path = os.path.join(target_dir, filename)
            
            # 保存图片
            img.save(file_path)
            
            return jsonify({'message': '文件上传成功', 'path': file_path}), 200
        else:
            return jsonify({'error': '图片中人脸数量不为1'}), 400
    
    return jsonify({'error': '未知错误'}), 500

if __name__ == '__main__':
    # 加载人脸库
    folder_path = '/Users/liuhaiyi/Library/CloudStorage/OneDrive-个人/haiyi文件/学校/个人/英才计划/2024寒假线下活动/工坊/person/'  # 更改为你的文件夹路径
    face_database = load_and_extract_features(folder_path)
    app.run(debug=True, host='0.0.0.0', port=27824)
