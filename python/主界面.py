from PyQt5 import QtWidgets, QtGui, QtCore
from ui import Ui_MainWindow
import datetime
import json
import os
import torch
from PIL import Image
from torchvision import transforms
from PyQt5.QtWidgets import *
from models.mobilenet import mobilenet  as create_model
from PyQt5 import QtCore
from plant_name import name_list
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)


        #日期更新

        # 设置表格自适应大小
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        #设置表格行列数
        self.tableWidget.setRowCount(15)
        self.tableWidget.setColumnCount(4)

        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        #列自适应，(水平方向占满窗体)
        self.tableWidget.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        #行自适应，(垂直方向占满窗体)

        #行列标题行隐藏

        #self.tableWidget.horizontalHeader().hide() #行
        self.tableWidget.verticalHeader().hide()  #列

        #设置表头
        self.tableWidget.setHorizontalHeaderLabels(['序号', '图像路径', '识别结果','置信度'])
        #显示网格线
        self.tableWidget.setShowGrid(True)


        #label置空
        self.label_res.setText('')
        self.label_pro.setText('')

        self.lineEdit_pic.setPlaceholderText('  点击选择图像')
        self.lineEdit_folder.setPlaceholderText('  点击选择文件夹')
        #加载检测模型

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        self.class_indict = json.load(json_file)

        # create model  创建模型网络
        self.model = create_model(class_num=67).to(self.device)
        # load model weights  加载模型
        model_weight_path = "weights/mobile_net.pth"
        self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
        self.model.eval()

        # 点击事件获取所选内容、行列
        self.tableWidget.cellPressed.connect(self.getPosContent)

        #交互按钮
        self.pushButton_pic.clicked.connect(self.select_img)  # 选择图像
        self.pushButton_pic_begin.clicked.connect(self.img_detect)  # 检测图像

        self.pushButton_folder.clicked.connect(self.choose_folder)  # 选择文件夹
        self.pushButton_folder_begin.clicked.connect(self.start_detect_folder)  # 检测文件夹

        self.timer_folder = QtCore.QTimer()
        self.timer_folder.timeout.connect(self.detect_folder_timer)



    def select_img(self):

        self.img_path, _ = QFileDialog.getOpenFileName(None, 'open img', '', "*.png;*.jpg;;All Files(*)")

        if self.img_path:
            print(self.img_path)
            self.lineEdit_pic.setText('  '+self.img_path)
            image = Image.open(self.img_path)
            r_image=image.resize((351,271))
            r_image.save('test.png')
            self.label_pic.setStyleSheet("image: url(./test.png)")  #将检测出的图片放到界面框中


    def img_detect(self):
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(1)
        self.folder_num = 0
        self.folder_res = []

        img_size = 224
        data_transform = transforms.Compose(
            [transforms.Resize(int(img_size * 1.143)),
             transforms.CenterCrop(img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # load image
        assert os.path.exists(self.img_path), "file: '{}' dose not exist.".format(self.img_path)
        img = Image.open(self.img_path)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # 调用模型进行检测
        with torch.no_grad():
            # predict class
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(self.class_indict[str(i)],
                                                      predict[i].numpy()))
        # 返回检测结果和准确率
        res = self.class_indict[str(list(predict.numpy()).index(max(predict.numpy())))]
        num = "%.2f" % (max(predict.numpy()) * 100) + "%"
        print(res, num)

        self.folder_res.append([str(self.folder_num + 1), self.img_path, name_list[int(res.split('_')[-1])], num])

        self.label_res.setText(name_list[int(res.split('_')[-1])])
        self.label_pro.setText(num)


        for column, data in enumerate(['1', self.img_path, name_list[int(res.split('_')[-1])], num]):
            self.tableWidget.setItem(0, column, QtWidgets.QTableWidgetItem(str(data)))



    def img_detect2(self,img_path):
        img_size = 224
        data_transform = transforms.Compose(
            [transforms.Resize(int(img_size * 1.143)),
             transforms.CenterCrop(img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # 调用模型进行检测
        with torch.no_grad():
            # predict class
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(self.class_indict[str(i)],
                                                      predict[i].numpy()))
        # 返回检测结果和准确率
        res = self.class_indict[str(list(predict.numpy()).index(max(predict.numpy())))]
        num = "%.2f" % (max(predict.numpy()) * 100) + "%"
        print(res, num)

        self.label_res.setText(name_list[int(res.split('_')[-1])])
        self.label_pro.setText(num)
        return name_list[int(res.split('_')[-1])], num

    def choose_folder(self):
        self.folder = QFileDialog.getExistingDirectory(None, "选择文件夹", "", QFileDialog.ShowDirsOnly)

        if self.folder:
            print(self.folder)
            self.lineEdit_folder.setText('  '+self.folder)


    def detect_folder(self):

        for num, path in enumerate(os.listdir(self.folder)):
            img_path = self.folder+'/'+path

            if img_path:
                print(img_path)
                image = Image.open(img_path)
                r_image = image.resize((351, 271))
                r_image.save('test.png')
                self.label_pic.setStyleSheet("image: url(./test.png)")  # 将检测出的图片放到界面框中



            res,pro = self.img_detect2(img_path)


            for column, data in enumerate([str(num+1),img_path,res,pro]):
                self.tableWidget.setItem(num, column, QtWidgets.QTableWidgetItem(str(data)))

    def start_detect_folder(self):

        if self.timer_folder.isActive() == False:  # 若定时器未启动

            self.tableWidget.clearContents()
            self.tableWidget.setRowCount(len(os.listdir(self.folder)))
            self.folder_list = [self.folder + '/' + path for path in os.listdir(self.folder)]
            self.folder_num = 0

            self.timer_folder.start(30)
            self.pushButton_folder_begin.setText('停止检测')

            self.folder_res=[]

        else:
            self.timer_folder.stop()
            self.pushButton_folder_begin.setText('开始检测')


    def detect_folder_timer(self):

        if self.folder_num==len(self.folder_list):
            self.timer_folder.stop()
            self.pushButton_folder_begin.setText('开始检测')

        else:
            img_path=self.folder_list[self.folder_num]

            print(img_path)

            image = Image.open(img_path)
            r_image = image.resize((351, 271))
            r_image.save('test.png')
            self.label_pic.setStyleSheet("image: url(./test.png)")  # 将检测出的图片放到界面框中
            res,pro = self.img_detect2(img_path)

            print(self.folder_num)
            self.folder_res.append([str(self.folder_num + 1), img_path, res, pro])

            for column, data in enumerate([str(self.folder_num + 1), img_path, res, pro]):
                self.tableWidget.setItem(self.folder_num, column, QtWidgets.QTableWidgetItem(str(data)))

            self.folder_num += 1



    # 获取选中行列、内容
    def getPosContent(self, row, col):
        try:
            content = self.tableWidget.item(row, col).text()
            print('选中内容:' + content)
            self.label_res.setText(self.folder_res[row][2])
            self.label_pro.setText(self.folder_res[row][3])
            image = Image.open(self.folder_res[row][1])
            print((self.folder_res[row][1]))
            r_image = image.resize((351, 271))
            r_image.save('test.png')
            self.label_pic.setStyleSheet("image: url(./test.png)")  # 将检测出的图片放到界面框中

        except:
            print('选中内容为空')



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())