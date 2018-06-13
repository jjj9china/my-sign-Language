# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:11:34 2018

@author: 15216
"""

# -*- coding:utf-8 -*-

"""
If you have a lot of pictures in your folder, 
but the order of numbers in the file names is chaotic, 
this small program can help you to rename
"""
import os

class ImageRename():
    def __init__(self):
        self.path = 'F:/Sign-Language-master/my_gestures/5'

    def rename(self):
        filelist = os.listdir(self.path)
        i = 1 # You can change the value of i to change the starting number of the name
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path),  format(str(i))+ '.jpg')
                os.rename(src, dst)
                #print ('converting %s to %s ...' % (src, dst))
                i = i + 1
        #print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()