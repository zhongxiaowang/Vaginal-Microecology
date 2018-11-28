# coding=utf-8
import os
import glob
import numpy as np
import cv2


class GetClassNumber(object):

    """
    Get the mult number of the imbalance labels
    """

    def __init__(self, read_path):
        self.read_path = read_path

    def get_the_number_of_every_class(self):
        nug3_numb, av2_numb, av3_numb = [0] * 3, [0] * 2, [0] * 3
        label_path = glob.glob(self.read_path + "*_nug3_label.png")
        for label_pa in label_path:
            nug3_label = cv2.imread(label_pa, cv2.IMREAD_GRAYSCALE)
            av2_label = cv2.imread(label_pa.replace("nug3", "av2"), cv2.IMREAD_GRAYSCALE)
            av3_label = cv2.imread(label_pa.replace("nug3", "av3"), cv2.IMREAD_GRAYSCALE)
            nug3_numb[np.argmax(nug3_label)] += 1
            av2_label[np.argmax(av2_label)] += 1
            av3_label[np.argmax(av3_label)] += 1
        return nug3_numb, av2_numb, av3_numb

    def get_mult_list(self):
        nug3_list, av2_list, av3_list = self.get_the_number_of_every_class()
        nug3_mult = np.max(nug3_list)/nug3_list
        av2_mult = np.max(av2_list) / av2_list
        av3_mult = np.max(av3_list) / av3_list
        return nug3_mult, av2_mult, av3_mult


class GetTrainList(object):

    """
    Get the 1:1:1:... train list
    """

    def __init__(self, read_label_path, read_image_path, list_save_path, train_path, nug3_mult, av2_mult, av3_mult):
        self.read_label_path = read_label_path
        self.read_image_path = read_image_path
        self.list_save_path = list_save_path
        self.train_path = train_path
        self.nug3_mult = nug3_mult
        self.av2_mult = av2_mult
        self.av3_mult = av3_mult

    def get_nug3_train_list(self):
        label_path = glob.glob(self.read_path + "*_nug3_label.png")
        train_nug3_list = []
        for label_pa in label_path:
            name = os.path.basename(label_pa).split("_")[0]
            nug3_label = cv2.imread(label_pa, cv2.IMREAD_GRAYSCALE)
            if np.argmax(nug3_label) == 0:
                for i in range(self.nug3_mult[0]):
                    train_nug3_list.append(self.train_path + name + ".png " +
                                           self.train_path + name + "_nug3_label.png" + "\n")
            elif np.argmax(nug3_label) == 1:
                for j in range(self.nug3_mult[1]):
                    train_nug3_list.append(self.train_path + name + ".png " +
                                           self.train_path + name + "_nug3_label.png" + "\n")
            elif np.argmax(nug3_label) == 2:
                for k in range(self.nug3_mult[2]):
                    train_nug3_list.append(self.train_path + name + ".png " +
                                           self.train_path + name + "_nug3_label.png" + "\n")

        with open(self.list_save_path + "train_nug3_list.txt", "w") as train_list:
            for ls in train_nug3_list:
                train_list.write(ls)

    def get_av2_train_list(self):
        label_path = glob.glob(self.read_path + "*_av2_label.png")
        train_av2_list = []
        for label_pa in label_path:
            name = os.path.basename(label_pa).split("_")[0]
            av2_label = cv2.imread(label_pa, cv2.IMREAD_GRAYSCALE)
            if np.argmax(av2_label) == 0:
                for i in range(self.av2_mult[0]):
                    train_av2_list.append(self.train_path + name + ".png " +
                                          self.train_path + name + "_av2_label.png" + "\n")
            elif np.argmax(av2_label) == 1:
                for j in range(self.av2_mult[1]):
                    train_av2_list.append(self.train_path + name + ".png " +
                                          self.train_path + name + "_av2_label.png" + "\n")

        with open(self.list_save_path + "train_av2_list.txt", "w") as train_list:
            for ls in train_av2_list:
                train_list.write(ls)

    def get_av3_train_list(self):
        label_path = glob.glob(self.read_path + "*_av3_label.png")
        train_av3_list = []
        for label_pa in label_path:
            name = os.path.basename(label_pa).split("_")[0]
            av3_label = cv2.imread(label_pa, cv2.IMREAD_GRAYSCALE)
            if np.argmax(av3_label) == 0:
                for i in range(self.av3_mult[0]):
                    train_av3_list.append(self.train_path + name + ".png " +
                                          self.train_path + name + "_av3_label.png" + "\n")
            elif np.argmax(av3_label) == 1:
                for j in range(self.av3_mult[1]):
                    train_av3_list.append(self.train_path + name + ".png " +
                                          self.train_path + name + "_av3_label.png" + "\n")
            elif np.argmax(av3_label) == 2:
                for k in range(self.av3_mult[2]):
                    train_av3_list.append(self.train_path + name + ".png " +
                                          self.train_path + name + "_av3_label.png" + "\n")

        with open(self.list_save_path + "train_av3_list.txt", "w") as train_list:
            for ls in train_av3_list:
                train_list.write(ls)


class GetTestList(object):

    """
    Get the test data list
    """

    def __init__(self, read_path, save_path, data_path):
        self.read_path = read_path
        self.save_path = save_path
        self.data_path = data_path

    def get_nugavtrivvcnug3_test_list(self):
        label_path = glob.glob(self.read_path + "*nug11_label.png")
        nug11, av8, tri, vvc, nug3 = [], [], [], [], []
        for label_pa in label_path:
            name = label_pa.split("/")[-1].split("_")[0]
            nug11.append(self.data_path + name + ".png" + " " + self.data_path + name + "_nug11_label.png" + "\n")
            av8.append(self.data_path + name + ".png" + " " + self.data_path + name + "_av8_label.png" + "\n")
            tri.append(self.data_path + name + ".png" + " " + self.data_path + name + "_tri_label.png" + "\n")
            vvc.append(self.data_path + name + ".png" + " " + self.data_path + name + "_vvc_label.png" + "\n")
            nug3.append(self.data_path + name + ".png" + " " + self.data_path + name + "_nug3_label.png" + "\n")

        with open(self.save_path + "nug11_test_list.txt", "w") as nug11_txt:
            for ls in nug11:
                nug11_txt.write(ls)

        with open(self.save_path + "av8_test_list.txt", "w") as av8_txt:
            for ls in av8:
                av8_txt.write(ls)

        with open(self.save_path + "tri_test_list.txt", "w") as tri_txt:
            for ls in tri:
                tri_txt.write(ls)

        with open(self.save_path + "vvc_test_list.txt", "w") as vvc_txt:
            for ls in vvc:
                vvc_txt.write(ls)

        with open(self.save_path + "nug3_test_list.txt", "w") as nug3_txt:
            for ls in nug3:
                nug3_txt.write(ls)

    def get_av23_list(self):
        label_path = glob.glob(self.read_path + "*av2_label.png")
        av2, av3 = [], []
        for label_pa in label_path:
            name = label_pa.split("/")[-1].split("_")[0]
            av2.append(self.data_path + name + ".png" + " " + self.data_path + name + "_av2_label.png" + "\n")
            av3.append(self.data_path + name + ".png" + " " + self.data_path + name + "_av3_label.png" + "\n")

        with open(self.save_path + "av2_test_list.txt", "w") as av2_txt:
            for ls in av2:
                av2_txt.write(ls)

        with open(self.save_path + "av3_test_list.txt", "w") as av3_txt:
            for ls in av3:
                av3_txt.write(ls)
