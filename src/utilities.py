import os
import tensorflow as tf
import cv2
import numpy as np
import skimage.color
import skimage.io
import skimage.transform


VGG_MEAN = np.array([123.68, 116.779, 103.939],
                    dtype='float32').reshape((1, 1, 3))  # RGB


def load_image(path, size=(200, 200)):
    """
    Taken from https://github.com/ry/tensorflow-vgg16/blob/master/tf_forward.py
    """
    # load image (skimage reads as RGB HWC [0,255] uint8)
    # convert to rgb if gray
    # VGG accepts BGR CHW [0,255] float32
    img = skimage.color.gray2rgb(skimage.io.imread(path)).astype('float32') / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize
    resized_img = skimage.transform.resize(crop_img, size)
    return resized_img  # output resized RGB [0,1]


def get_immediate_subfiles(a_dir):
    return sorted([os.path.join(a_dir, name) for name in os.listdir(a_dir)
                   if os.path.isfile(os.path.join(a_dir, name))])


def load_images(path, size=(40, 200, 200)):
    images = []
    count = 0
    for frame_path in get_immediate_subfiles(path):
        if count == size[0]:
            break
        img = load_image(frame_path, size[1:])
        images.append(img)
        count += 1
    return np.array(images)


def check_snapshots(run_id):
    snapshots_folder = 'snapshots/' + run_id + '/'
    logs_folder = 'logs/' + run_id + '/'

    checkpoint = tf.train.latest_checkpoint(snapshots_folder)

    resume = False
    start_iteration = 0

    if checkpoint:
        choice = ''
        while choice != 'y' and choice != 'n':
            print 'Snapshot file detected (' + checkpoint + \
                  ') would you like to resume? (y/n)'
            choice = raw_input().lower()

            if choice == 'y':
                resume = checkpoint
                start_iteration = int(checkpoint.split(snapshots_folder)
                                      [1][5:-5])
                print 'resuming from iteration ' + str(start_iteration)
            else:
                print 'removing old snapshots and logs, training from scratch'
                resume = False
                for file in os.listdir(snapshots_folder):
                    if file == '.gitignore':
                        continue
                    os.remove(snapshots_folder + file)
                for file in os.listdir(logs_folder):
                    if file == '.gitignore':
                        continue
                    os.remove(logs_folder + file)
    else:
        print "No snapshots found, training from scratch"

    return resume, start_iteration


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    v, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang*(180/np.pi/2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def vgg_process(image):
    image = image * 255.0  # [0,1] -> [0,255]
    image = image - VGG_MEAN  # mean subtract
    image = image[..., ::-1]  # RGB -> BGR
    return image


def vgg_deprocess(image, no_clip=False, unit_scale=False):
    image = image[..., ::-1]  # BGR -> RGB
    image = image + VGG_MEAN
    if not no_clip:
        image = np.clip(image, 0, 255).astype('uint8')
    if unit_scale:
        image = image / 255.0
    return image


def get_bounds(images, im_size, frame_count):
    '''
    Helper function to get optimisation bounds from source image.
    Taken from Gatys' DeepTextures repo.
    :param images: a list of images
    :param im_size: image size (height, width) for the generated image
    :return: list of bounds on each pixel for the optimisation
    '''
    lowerbound = np.min([im.min() for im in images])
    upperbound = np.max([im.max() for im in images])
    bounds = list()
    for b in range(frame_count * im_size[0] * im_size[1] * 3):
        bounds.append((lowerbound, upperbound))
    return np.asarray(bounds)


def gauss2d_kernel(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
