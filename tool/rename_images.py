# -*- coding:utf8 -*-
import os
class BatchRename():
    def __init__(self):
        self.path = 'C:/carlos/image_download/supplement_facts_image'
    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        group = 0
        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.JPG') or item.endswith('.jpeg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '01'+ str("%05d" % i) + '.jpg')
                with open('C:/carlos/image_download/supplement_facts_image.txt', 'a') as f:
                    f.write('data/nutrition_facts_images/'+'01'+ str("%05d" % i) + '.jpg'+' 1'+'\n')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s' % (src, dst))
                    group = 1
                except:
                    continue
            if item.endswith('.png'):
                # continue
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '01'+ str("%05d" % i) + '.png')
                with open('C:/carlos/image_download/supplement_facts_image.txt', 'a') as f:
                    f.write('data/nutrition_facts_images/'+'01'+ str("%05d" % i) + '.png'+' 1'+'\n')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s' % (src, dst))
                    group = 1
                except:
                    continue
            if group:
                i = i + 1
        print('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()