import skimage.color
import skimage.io
import skimage.transform
import scipy.misc
import subprocess
import os
import numpy as np

def load_image(path, size=(200, 200)):
    """
    Taken from https://github.com/ry/tensorflow-vgg16/blob/master/tf_forward.py
    """
    # load image (skimage reads as RGB HWC [0,255] uint8)
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


def get_immediate_subdirectories(a_dir):
    return sorted([os.path.join(a_dir, name) for name in os.listdir(a_dir)
                   if os.path.isdir(os.path.join(a_dir, name))])


def get_immediate_subfiles(a_dir):
    return sorted([os.path.join(a_dir, name) for name in os.listdir(a_dir)
                   if os.path.isfile(os.path.join(a_dir, name))])


# TODO: since this is for MSOEnet (which used cv2.imread,
#       which reads as BGR HWC [0,255] uint8), images should
#       be read in grayscale HWC [0,1] float32
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


def makegif(path):
    try:
        os.makedirs(path + '/cropped')
    except OSError:
        if not os.path.isdir(path + '/cropped'):
            raise
    imgs = load_images(path, size=(12,256,256))
    for i in range(imgs.shape[0]):
        scipy.misc.toimage(imgs[i], cmin=0.0, cmax=1.0).save(path + '/cropped' + '/frame_' + '%08d' % i + '.png')
    output = path.split('/')
    subprocess.call(['./makegif.sh', path + '/cropped' + '/frame*', path + '/cropped' + '/' + output[1] + '.gif'])


def makegifs():
    for folder in get_immediate_subdirectories('.'):
        makegif(folder)

makegifs()
