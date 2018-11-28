# coding=utf-8
import os
import glob
import numpy as np
import cv2
from label_process import GetData, GetNugAvTriVvcLabel, GetNug3Av23
from list_process import GetClassNumber, GetTestList, GetTrainList

PATH_ALL = '/Users/andy/Documents/cg_data6(test)/'
SAVE_DATA = '/Users/andy/Documents/test_data/'
NUG_MASK = '/Users/andy/Desktop/make_mask_nug/'
AV_MASK = '/Users/andy/Desktop/make_mask_av/'
TRI_MASK = '/Users/andy/Desktop/make_mask_dch/'
SAVE_LABEL = '/Users/andy/Documents/data/all_test_label/'
DATA_PATH = '/storage2/zxwang/new_2010/test_data/'

# a = GetData(PATH_ALL, SAVE_DATA)
# a.get_all_data()

# b = GetNugAvTriVvcLabel(NUG_MASK, AV_MASK, TRI_MASK, SAVE_DATA, SAVE_LABEL)
# b.save_data()

# c = GetNug3Av23(SAVE_LABEL, SAVE_LABEL)
# c.get_nug3()
# c.get_av23()

d = GetTestList(SAVE_LABEL, SAVE_LABEL, DATA_PATH)
#d.get_nugavtrivvcnug3_test_list()
d.get_av23_list()
