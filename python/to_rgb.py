from PIL import Image
import os

#数据集格式类型命名统一
for i in os.listdir('all_data'):
    num=1
    for file_name in os.listdir('all_data/{}'.format(i)):
        img =Image.open('all_data/{}/'.format(i) + file_name)
        print(file_name)
        if img.mode != "RGB" :
            print(file_name)
            img_rgb = img.convert("RGB")
            img_rgb.save('all_data/{}/{}.png'.format(i, num))
            os.remove('all_data/{}/'.format(i) + file_name)

        else:
            img.save('all_data/{}/{}.png'.format(i, num))
            os.remove('all_data/{}/'.format(i) + file_name)


        num+=1