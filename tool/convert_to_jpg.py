# -*- coding:utf8 -*-
import os
from PIL import Image

class BatchRename():
    def __init__(self):
        self.path = 'C:/carlos/image_download/caution'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.bmp') or item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), item[:-4] + '.jpg')
                img = Image.open(src)
                img = img.convert('RGB')
                img.save(dst, format='jpeg', quality=100)
                img.close()
                group = 1
                # src = os.path.join(os.path.abspath(self.path), item)
                # dst = os.path.join(os.path.abspath(self.path), '01'+ str("%05d" % i) + '.jpg')
                # with open('C:/carlos/image_download/data.txt', 'a') as f:
                #     f.write('data/nutrition_facts_images/'+'02'+ str("%05d" % i) + '.jpg'+' 2'+'\n')
                # try:
                #     # os.rename(src, dst)
                #     print('converting %s to %s' % (src, dst))
                #     group = 1
                # except:
                #     continue
            if item.endswith('.jpg'):
                group = 0
                # src = os.path.join(os.path.abspath(self.path), item)
                # dst = os.path.join(os.path.abspath(self.path), '01'+ str("%05d" % i) + '.png')
                # with open('C:/carlos/image_download/data.txt', 'a') as f:
                #     f.write('data/nutrition_facts_images/'+'02'+ str("%05d" % i) + '.png'+' 2'+'\n')
                # try:
                #     #os.rename(src, dst)
                #     print('converting %s to %s' % (src, dst))
                #     group = 1
                # except:
                #     continue
            if group:
                i = i + 1
        print('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()