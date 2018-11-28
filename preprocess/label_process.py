# coding=utf-8
import os
import glob
import numpy as np
import cv2


class GetData(object):

    """
    Get all useful 1:1 images and labels, and save them.
    """

    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.image_lists = glob.glob(data_path + "*/*.JPG")
        self.label_lists = glob.glob(data_path + "*/*.PNG")

    def get_number_set(self):
        label_set, image_set = set(), set()
        image_path, label_path = self.image_lists, self.label_lists
        for label_pt in label_path:
            label_set.add(label_pt.split('/')[-2])
        for image_pt in image_path:
            image_set.add(image_pt.split('/')[-2])
        return sorted(list(label_set & image_set))

    def get_lists_in_number(self):
        image_ls, label_ls = [], []
        image_number = self.get_number_set()
        print("total data %d" % len(image_number))
        for numb in image_number:
            image_pa = glob.glob(self.data_path + numb + "/*.JPG")
            if len(image_pa) >= 2:
                print("image", numb)
            image_ls.append(image_pa)
            label_pa = glob.glob(self.data_path + numb + "/*.PNG")
            if len(label_pa) >= 2:
                print("label", numb)
            label_ls.append(label_pa)
        return image_ls, label_ls

    def save_data(self, data):
        if len(data) >= 3:
            print(data)
        elif len(data) == 2:
            print("2 data")
            data1 = cv2.imread(data[0], cv2.IMREAD_COLOR)
            data2 = cv2.imread(data[1], cv2.IMREAD_COLOR)
            if (data1 == data2).all():
                if data[0].split(".")[-1] == "JPG":
                    cv2.imwrite(self.save_path + data[0].split('/')[-2] + ".png", data1)
                elif data[0].split(".")[-1] == "PNG":
                    cv2.imwrite(self.save_path + data[0].split('/')[-2] + "_label.png", data1)
                else:
                    raise IOError
            else:
                print("wrong image data", data)
        elif len(data) == 1:
            data1 = cv2.imread(data[0], cv2.IMREAD_COLOR)
            if data[0].split(".")[-1] == "JPG":
                cv2.imwrite(self.save_path + data[0].split('/')[-2] + ".png", data1)
            elif data[0].split(".")[-1] == "PNG":
                cv2.imwrite(self.save_path + data[0].split('/')[-2] + "_label.png", data1)
            else:
                raise IOError
        else:
            print("wrong data path")

    def get_all_data(self):

        image_all_ls, label_all_ls = self.get_lists_in_number()
        image_num, label_num = 0, 0
        for image in image_all_ls:
            self.save_data(image)
            image_num += 1
            if image_num % 10 == 0:
                print("%d images has been finished" % image_num)
        for label in label_all_ls:
            self.save_data(label)
            label_num += 1
            if label_num % 10 == 0:
                print("%d labels has been finished" % label_num)


class GetNugAvTriVvcLabel(object):

    """
    Get all labels including: Nugent(0-10), Av(0-6[and not given]), Trichomonas(0-1), Vvc(0-1)
    """

    def __init__(self, nug_mask_path, av_mask_path, tri_mack_path, read_path, save_path):
        self.nug_mask_path = nug_mask_path
        self.av_mask_path = av_mask_path
        self.tri_mask_path = tri_mack_path
        self.read_path = read_path
        self.save_path = save_path

    def get_nug_mask(self):
        mask = [[]] * 11
        for ii in range(11):
            nug_mask = cv2.imread(self.nug_mask_path + str(ii) + ".PNG")
            mask[ii] = nug_mask[597:608, 168:182, 0]
        return mask

    def get_av_mask(self):
        mask1 = [[]] * 8
        for ij in range(8):
            av_mask = cv2.imread(self.av_mask_path + str(ij) + ".PNG")
            mask1[ij] = av_mask[639:652, 121:135, 0]
        return mask1

    def get_tri_mask(self):
        mask2 = [[]] * 2
        for ik in range(2):
            tri_mask = cv2.imread(self.tri_mask_path + str(ik) + ".PNG")
            mask2[ik] = tri_mask[440:460, 245:275, 0]
        return mask2

    def save_data(self):
        label_path = glob.glob(self.read_path + "/*_label.png")
        num = 0
        for label_ph in label_path:
            num += 1
            nug_label_mask, av_label_mask = np.zeros([1, 11]), np.zeros([1, 8])
            tri_label_mask, vvc_label_mask = np.zeros([1, 2]), np.zeros([1, 2])
            name = os.path.basename(label_ph)
            if os.path.exists(self.save_path + name.replace("label.png", "nug11_label.png")):
                continue
#            image = cv2.imread(label_ph.replace("_label.png", ".png"), cv2.IMREAD_COLOR)
            label = cv2.imread(label_ph, cv2.IMREAD_COLOR)
            cv2.imwrite(self.save_path + name, label)

            for i in range(11):
                if (self.get_nug_mask()[i] == label[597:608, 168:182, 0]).all():
                    nug_label_mask[0][i] = 255
                else:
                    continue
            cv2.imwrite(self.save_path + name.replace("label.png", "nug11_label.png"), nug_label_mask)

            for j in range(8):
                if (self.get_av_mask()[j] == label[639:652, 121:135, 0]).all():
                    av_label_mask[0][j] = 255
                else:
                    continue
            cv2.imwrite(self.save_path + name.replace("label.png", "av8_label.png"), av_label_mask)

            for k in range(2):
                if (self.get_tri_mask()[k] == label[440:460, 245:275, 0]).all():
                    tri_label_mask[0][k] = 255
                else:
                    continue
            cv2.imwrite(self.save_path + name.replace("label.png", "tri_label.png"), tri_label_mask)

            if np.sum(np.uint8(label[485:505, 240:275, 0])) == 169065 or \
                    np.sum(np.uint8(label[560:580, 240:275, 0])) == 169065:
                vvc_label_mask[0][1] = 255
            else:
                vvc_label_mask[0][0] = 255
            cv2.imwrite(self.save_path + name.replace("label.png", "vvc_label.png"), vvc_label_mask)
            if np.sum(nug_label_mask) != 255 or np.sum(av_label_mask) != 255 or np.sum(tri_label_mask) != 255 \
                    or np.sum(vvc_label_mask) != 255:
                print label_ph
                raise ValueError
            if num % 20 == 0:
                print("%d labels have been finished" % num)


class GetNug3Av23(object):

    """
    Get the labels of Nugent(0-2), Av(0-1), Av(0-2)
    """

    def __init__(self, read_path, save_path):
        self.read_path = read_path
        self.save_path = save_path

    def get_nug3(self):
        label_nug_path = glob.glob(self.read_path + "*nug11_label.png")
        num = 0
        for label_nug_pa in label_nug_path:
            nug_mask = np.zeros([1, 3])
            nug_label = cv2.imread(label_nug_pa, cv2.IMREAD_GRAYSCALE)
            if 0 <= np.argmax(nug_label) <= 3:
                nug_mask[0][0] = 255
            elif 4 <= np.argmax(nug_label) <= 6:
                nug_mask[0][1] = 255
            elif 7 <= np.argmax(nug_label) <= 10:
                nug_mask[0][2] = 255
            else:
                print(label_pa)
                raise ValueError
            cv2.imwrite(label_nug_pa.replace("nug11", "nug3"), nug_mask)
            num += 1
            if num % 50 == 0:
                print("%d labels have finished" % num)

    def get_av23(self):
        label_av_path = glob.glob(self.read_path + "*av8_label.png")
        num, sum1 = 0, 0
        for label_av_pa in label_av_path:
            av_mask1 = np.zeros([1, 2])
            av_mask2 = np.zeros([1, 3])
            av_label = cv2.imread(label_av_pa, cv2.IMREAD_GRAYSCALE)
            if 0 <= np.argmax(av_label) <= 2:
                av_mask1[0][0] = 255
                av_mask2[0][0] = 255
            elif 3 <= np.argmax(av_label) <= 4:
                av_mask1[0][1] = 255
                av_mask2[0][1] = 255
            elif 5 <= np.argmax(av_label) <= 6:
                av_mask1[0][1] = 255
                av_mask2[0][2] = 255
            else:
                sum1 += 1
                continue
            cv2.imwrite(label_av_pa.replace("av8", "av2"), av_mask1)
            cv2.imwrite(label_av_pa.replace("av8", "av3"), av_mask2)
            num += 1
            if num % 50 == 0:
                print("%d labels have finished" % num)
        print("%d images have no label" % sum1)
