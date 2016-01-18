import numpy as np


def get_intersection_bbox(bbox1, bbox2, image_width, image_height):
    l1, t1, w1, h1 = bbox1
    r1 = l1 + w1
    b1 = t1 + h1
    l2, t2, w2, h2 = bbox2
    r2 = l2 + w2
    b2 = t2 + h2

    l = max(l1, l2)
    r = min(r1, r2)
    t = max(t1, t2)
    b = min(b1, b2)

    l = max(0, l)
    r = min(r, image_width - 1)
    t = max(0, t)
    b = min(b, image_height - 1)

    w = r - l
    h = b - t

    w = max(0, w)
    h = max(0, h)

    return np.array([l, t, w, h])


def get_union_bbox(bbox1, bbox2, image_width, image_height):
    l1, t1, w1, h1 = bbox1
    r1 = l1 + w1
    b1 = t1 + h1
    l2, t2, w2, h2 = bbox2
    r2 = l2 + w2
    b2 = t2 + h2

    l = min(l1, l2)
    r = max(r1, r2)
    t = min(t1, t2)
    b = max(b1, b2)

    l = max(0, l)
    r = min(r, image_width - 1)
    t = max(0, t)
    b = min(b, image_height - 1)

    w = r - l
    h = b - t

    w = max(0, w)
    h = max(0, h)

    return np.array([l, t, w, h])


def get_bbox_area(bbox):
    l, t, w, h = bbox
    return w * h


def get_iou(bbox1, bbox2, image_width, image_height):
    bbox_intersection = get_intersection_bbox(bbox1, bbox2, image_width, image_height)
    bbox_union = get_union_bbox(bbox1, bbox2, image_width, image_height)

    area_intersection = get_bbox_area(bbox_intersection)
    area_union = get_bbox_area(bbox_union)

    assert area_union >= 0
    assert area_intersection >= 0

    if area_union == 0:
        return 0

    return area_intersection / float(area_union)
