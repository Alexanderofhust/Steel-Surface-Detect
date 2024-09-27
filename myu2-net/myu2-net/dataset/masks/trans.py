from PIL import Image
import os


path1 = 'E:\\ai\\myu2-net\\dataset\\masks\\test'
path2 = 'E:\\ai\\myu2-net\\dataset\\masks\\train'

for file in os.listdir(path2):
    if file.endswith('.png'):
        img = Image.open(os.path.join(path2, file))
        file_name = os.path.splitext(file)[0]
        img.save(os.path.join(path2, file_name+'.jpg'))