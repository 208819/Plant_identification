import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.mobilenet import mobilenet  as create_model


def main(img_path):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.143)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model  创建模型网络
    model = create_model(class_num=67).to(device)
    # load model weights  加载模型
    model_weight_path = "weights/plant-best-epoch.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    #调用模型进行检测
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()


    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    # 返回检测结果和准确率
    res = class_indict[str(list(predict.numpy()).index(max(predict.numpy())))]
    num= "%.2f" % (max(predict.numpy()) * 100) + "%"
    print(res,num)
    return res,num


if __name__ == '__main__':
    img_path = r"all_data\plant_2\3.png"
    main(img_path)
