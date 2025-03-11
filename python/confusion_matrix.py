import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import read_split_data
from models.mobilenet import mobilenet  as create_model
import numpy as np
from sklearn.metrics import confusion_matrix
from plant_name import name_list
def main(img_path,model):

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.143)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    #调用模型进行检测
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print(int(predict_cla))
        return int(predict_cla)



if __name__ == '__main__':
    import matplotlib.font_manager

    # 设置字体路径
    font_path = 'font/simsun.ttc'  # 替换为实际的字体文件路径
    font = matplotlib.font_manager.FontProperties(fname=font_path)
    #读取所有数据图像,并可以设置划分比例
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data('./all_data',
                                                                                               val_rate=0.2)
    print(len(train_images_path))
    print(train_images_label)


    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model  创建模型网络

    model = create_model(class_num=67).to(device)
    # load model weights  加载模型
    model_weight_path = "weights/mobile_net.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    res_label=[]
    for path in train_images_path:
        res_pre = main(path,model)
        res_label.append(res_pre)
    print('===='*10)
    print(train_images_label)
    print(res_label)



    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(classes)))
        plt.xticks(xlocations, classes, rotation=45,fontproperties=font)
        plt.yticks(xlocations, classes,fontproperties=font)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    classes=[name_list[int(i.split('_')[-1])] for i in os.listdir('all_data')]
    num_classes = len(classes)
    tick_marks = np.array(range(len(classes))) + 0.5

    y_true=np.array(train_images_label)
    y_pred=np.array(res_label)
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)  # 设置精度
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 各类比率
    print('混淆矩阵：')
    print(cm_normalized)
    accuracy = np.mean([cm_normalized[i, i] for i in range(num_classes)])  # 混淆矩阵右斜线上的结果之和的均值即为准确率
    print('准确率：' + str(round(accuracy, 2)))

    # 创建窗口，绘制混淆矩阵图
    plt.figure(figsize=(20, 18), dpi=220)
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)

    # 添加每格分类比率结果
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
    # 设置图表
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(False, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # 显示绘图
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix '+ 'acc:' + str(round(accuracy, 2)))
    plt.savefig('confusion_matrix_mobile_net.png', format='png')  # 保存结果
    plt.show()  # 显示窗口













