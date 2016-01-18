import math
from skimage.transform import SimilarityTransform, warp, rotate


def get_head_crop(img, pt1, pt2):
    im = img.copy()
    minh = 10
    minw = 20

    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    dist = math.hypot(x, y)
    croph = int((im.shape[0] - 1.0 * dist) // 2)
    cropw = int((im.shape[1] - 2.0 * dist) // 2)
    newh = im.shape[0] - 2 * croph
    neww = im.shape[1] - 2 * cropw

    if croph <= 0 or cropw <= 0 or newh < minh or neww < minw:
        return im
    else:
        angle = math.atan2(y, x) * 180 / math.pi
        centery = 0.4 * pt1[1] + 0.6 * pt2[1]
        centerx = 0.4 * pt1[0] + 0.6 * pt2[0]
        center = (centerx, centery)
        im = rotate(im, angle, resize=False, center=center)
        imcenter = (im.shape[1] / 2, im.shape[0] / 2)
        trans = (center[0] - imcenter[0], center[1] - imcenter[1])
        tform = SimilarityTransform(translation=trans)
        im = warp(im, tform)
        im = im[croph:-croph, cropw:-cropw]
        return im
