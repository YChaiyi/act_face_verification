# 项目简介

这是一个基于 Flask 和深度学习的面部识别与图像处理 API 服务，主要用于面部识别、颜色检测、人脸检测及验证等功能。本项目集成了 MTCNN 和 InceptionResnetV1 模型进行人脸检测与特征提取，并利用 dlib 库进行嘴巴开合检测。该系统支持图片上传、人员识别、颜色匹配以及人脸验证等多种功能，适用于各种与面部识别相关的应用场景。

> **注意**：此项目适用于教育或研究用途，如有部署和技术疑问请提交 issue。使用本代码时请尊重开源协议。

# 部署须知
1. **Python 环境**：需 Python 3.x 版本，推荐使用 Python 3.12 或以上。
2. **依赖库**：
   - Flask
   - Flask-CORS
   - Pillow
   - NumPy
   - OpenCV
   - dlib
   - facenet-pytorch
   - uuid
3. **文件夹结构**：
   - 上传的图片存储在 `uploaded_images` 文件夹中。
   - 训练人脸特征的数据库文件夹路径需要自定义。

# 结构
```markdown
face_recognition_project
├── app.py  # 主程序入口，包含所有的 API 路由
├── uploaded_images  # 存储上传的图片
├── face_database  # 存储训练后的面部特征数据
└── requirements.txt  # 项目依赖列表
```

# 技术总结

## 后端

### 1. 技术栈
- **Flask**：轻量级的 Python Web 框架，提供了 RESTful API 服务。
- **MTCNN**：用于人脸检测和对齐。
- **InceptionResnetV1**：用于从对齐的人脸图像中提取特征向量，进行人脸识别。
- **dlib**：用于嘴巴开合检测，判断是否为真实人物。
- **OpenCV**：用于图像处理和颜色分析。
- **Pillow**：处理图像的基本操作（如加载和转换格式）。
- **UUID**：生成唯一的文件名，以确保每次上传的文件不会覆盖。

### 2. 功能模块

#### 2.1 人脸识别
- **上传人脸并识别** (`/find_name`): 接收上传的图片，识别其中的人脸并返回匹配的身份。
- **加载人脸数据库**：支持通过指定文件夹路径加载预先存储的人脸特征，进行比对识别。

#### 2.2 颜色检测
- **检查颜色相似度** (`/check_color`): 接收颜色名称和图片，检测人脸区域的平均颜色并与指定颜色进行比对，判断是否相似。

#### 2.3 人脸特征与验证
- **检查嘴巴是否张开** (`/check_mouth_open`): 检测上传图片中的嘴巴是否张开，用于判断是否为真实人物。
  
#### 2.4 文件上传
- **文件上传** (`/upload`): 上传包含人脸的图像并保存到指定目录中，支持多种格式（如 JPG、PNG）。上传时会自动进行人脸检测，确保图像中只有一个人脸。

### 3. 安全性
- **CORS**：通过 `Flask-CORS` 库实现跨域资源共享，确保前端与后端能够顺利通信。
- **日志**：使用 Python 的 `logging` 库记录重要操作，便于调试和追踪问题。

### 4. 数据存储
- **人脸数据库**：通过加载存储在本地文件夹中的图像，提取特征向量并存储在字典中以供识别时进行比对。
- **上传图片**：上传的图片保存在 `uploaded_images` 文件夹中，每个文件都以唯一的 UUID 命名，避免冲突。

## 前端
该项目目前主要提供 API 服务，前端开发可根据需求集成 API 接口。在前端集成时，可以使用 JavaScript 的 `fetch` 或 `Axios` 来与后端进行数据交互。

### 1. 请求示例
- **上传文件**：
  ```js
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('name', 'person_name');

  fetch('/upload', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => console.log(data));
  ```

### 2. API 示例

- **识别人脸**：
  ```bash
  POST /find_name
  Content-Type: multipart/form-data
  { "file": "image.jpg" }
  ```
- **检查颜色相似度**：
  ```bash
  POST /check_color
  Content-Type: application/x-www-form-urlencoded
  { "color": "red", "image": "image.jpg" }
  ```

# 安装与配置

### 1. 克隆项目
```bash
git clone https://github.com/your-repo/face-recognition.git
cd face-recognition
```

### 2. 安装依赖
创建虚拟环境并安装所需的依赖：
```bash
python3 -m venv venv
source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置人脸数据库
将用于训练的人脸图像放入指定目录，并修改 `app.py` 中的 `folder_path` 为该目录路径。

### 4. 运行服务
```bash
python app.py
```

服务默认运行在 `http://localhost:27824`，可以根据需要修改配置。
