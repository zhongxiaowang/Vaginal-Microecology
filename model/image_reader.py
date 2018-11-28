import os
import numpy as np
import random
import tensorflow as tf


def random_adjust_brightness(image, max_delta=0.2, seed=None):
    """Randomly adjusts brightness. """
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
    image = tf.image.adjust_brightness(image / 255, delta) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25, seed=None):
    """Randomly adjusts contrast. """
    contrast_factor = tf.random_uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_contrast(image / 255, contrast_factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


def random_adjust_hue(image, max_delta=0.02, seed=None):
    """Randomly adjusts hue. """
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
    image = tf.image.adjust_hue(image / 255, delta) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25, seed=None):
    """Randomly adjusts saturation. """
    saturation_factor = tf.random_uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_saturation(image / 255, saturation_factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


def image_jittering(image):
    """Randomly distorts color.

    Randomly distorts color using a combination of brightness, hue, contrast and
    saturation changes. Makes sure the output image is still between 0 and 255.

    Args:
        image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
                     with pixel values varying between [0, 255].

    Returns:
        image: image which is the same shape as input image.
    """

    image = random_adjust_brightness(image, max_delta=0.08)
    image = random_adjust_saturation(image, min_delta=0.92, max_delta=1.08)
    image = random_adjust_hue(image, max_delta=0.08)
    image = random_adjust_contrast(image, min_delta=0.92, max_delta=1.08)

    return image


def image_scaling(img, label):

    """
    Randomly scales the images between 0.5 to 1.25 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random_uniform(shape=[1], minval=1, maxval=1.5, dtype=tf.float32)
    input_shape = tf.shape(img)
    input_shape_float = tf.to_float(input_shape)
    scaled_input_shape = tf.to_int32(tf.round(input_shape_float * scale))
    
    img = tf.image.resize_images(img, scaled_input_shape)
    label = tf.image.resize_images(label, scaled_input_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    img = tf.image.resize_image_with_crop_or_pad(img, input_shape)
    label = tf.image.resize_image_with_crop_or_pad(label, input_shape)

    return img, label


def image_mirroring(img, label):

    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    random_var1 = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    img = tf.cond(pred=tf.equal(random_var1, 0), true_fn=lambda: tf.image.flip_left_right(img), false_fn=lambda: img)
    label = tf.cond(pred=tf.equal(random_var1, 0), true_fn=lambda: tf.image.flip_left_right(label), false_fn=lambda: label)

    random_var2 = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    img = tf.cond(pred=tf.equal(random_var2, 0), true_fn=lambda: tf.image.flip_up_down(img), false_fn=lambda: img)
    label = tf.cond(pred=tf.equal(random_var2, 0), true_fn=lambda: tf.image.flip_up_down(label), false_fn=lambda: label)
        
    random_var3 = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    img = tf.cond(pred=tf.equal(random_var3, 0), true_fn=lambda: tf.image.rot90(img, 1), false_fn=lambda: img)
    label = tf.cond(pred=tf.equal(random_var3, 0), true_fn=lambda: tf.image.rot90(label, 1), false_fn=lambda: label)

    random_var4 = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    img = tf.cond(pred=tf.equal(random_var4, 0), true_fn=lambda: tf.image.rot90(img, 2), false_fn=lambda: img)
    label = tf.cond(pred=tf.equal(random_var4, 0), true_fn=lambda: tf.image.rot90(label, 2), false_fn=lambda: label)

    random_var5 = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    img = tf.cond(pred=tf.equal(random_var5, 0), true_fn=lambda: tf.image.rot90(img, 3), false_fn=lambda: img)
    label = tf.cond(pred=tf.equal(random_var5, 0), true_fn=lambda: tf.image.rot90(label, 3), false_fn=lambda: label)

    return img, label


def read_labeled_image_list(data_list):

    """
    Reads txt file containing paths to images and ground truth masks.

    Args:
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """

    f = open(data_list, 'r')
    images = []
    labels = []
    names = []
    for line in f:
        try:
            if len(line.strip("\n").split(' ')) == 2:
                image, label = line.strip("\n").split(' ')
            elif len(line.strip("\n").split(' ')) == 1:
                image = line.strip("\n")
                label = ''
            name = image.split('.')[0]
        except ValueError:  # Adhoc for test.
            print 'List file format wrong!'
            exit(1)
        images.append(image)
        labels.append(label)
        names.append(name)
    return images, labels, names


def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, random_jittering, img_mean):

    """
    Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      img_mean: vector of mean colour values.

    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    name_contents = input_queue[2]

    img = tf.image.decode_png(img_contents, channels=3)
    label = tf.image.decode_png(label_contents, channels=1)

    img = tf.cast(img, dtype=tf.float32)

    # Extract mean.
    img -= img_mean

    if input_size is not None:
        w, h = input_size
        img = tf.image.resize_images(img, tf.convert_to_tensor([w, h]))
        label = tf.image.resize_images(label, tf.convert_to_tensor([w, h]))

        # Randomly scale the images and labels.
        if random_scale:
            img, label = image_scaling(img, label)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = image_mirroring(img, label)

        # Randomly jittering the images and labels.
        if random_jittering:
            img = image_jittering(img)

    return img, label, tf.convert_to_tensor(name_contents, dtype=tf.string)


class ImageReader(object):

    """
    Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    """

    def __init__(self, data_list, input_size, random_scale, random_mirror, random_jittering, img_mean, coord):

        """
        Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          random_jittering: whether to randomly jitter the images.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator.
        """

        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list, self.name_list = read_labeled_image_list(self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.names = tf.convert_to_tensor(self.name_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels, self.names],
                                                   shuffle=input_size is not None)  # not shuffling if it is val
        self.image, self.label, self.img_name = read_images_from_disk(self.queue, self.input_size, random_scale,
                                                                      random_mirror, random_jittering, img_mean)

    def dequeue(self, num_elements):

        """
        Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.
        """

        image_batch, label_batch, name_batch = \
            tf.train.shuffle_batch([self.image, self.label, self.img_name], num_elements, 128, 64)
        return image_batch, label_batch, name_batch
