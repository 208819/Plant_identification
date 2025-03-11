from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import os
from models.mobilenet import mobilenet as create_model
import json
import base64
import io
from plant_name import name_list
app = Flask(__name__)

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

with open(json_path, "r") as json_file:
    class_indict = json.load(json_file)

# 创建模型网络
model = create_model(class_num=67).to(device)
# 加载模型权重
model_weight_path = "weights/mobile_net.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

# 图像预处理
def preprocess_image(image):
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.143)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = data_transform(image)
    img = torch.unsqueeze(img, dim=0)
    return img

# 检测接口
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    try:
        # 解码Base64图像
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))

        # 预处理图像
        img = preprocess_image(image)

        # 调用模型进行检测
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # 获取检测结果和置信度
        res = class_indict[str(predict_cla)]
        confidence = float(predict[predict_cla].numpy())

        # 返回结果
        return jsonify({
            'plant_name': name_list[int(res.split('_')[-1])],
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)