import numpy as np

# numpy post-process
def yolo_postprocess_(outs, anchor_path, class_path, image_h, image_w, input_shape, score_thres, nms_thres, keep_aspect_ratio,
                      is_v5=False, sigmoid_in_post_v5=False):
    with open(anchor_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    # print(anchors, anchors.size)
    with open(class_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    # print(len(class_names))
    num_class = len(class_names)

    step = 32
    out_list = []
    stride = []

    for out in outs:
        # (batch, h, w, c)
        a = np.reshape(out, (1, input_shape[1]//step, input_shape[0]//step, 3, 5 + num_class))
        out_list.append(a)
        stride.append(step)
        step = int(step / 2)

    boxes, scores, classes = yolo_out(out_list, anchors, image_h, image_w, input_shape, score_thres, nms_thres, keep_aspect_ratio,
                                      is_v5, sigmoid_in_post_v5, stride)

    # To public field
    new_bboxes = []

    if (boxes is not None):
        bboxes = list(map(list, zip(*[boxes.tolist(), scores.tolist(), classes.tolist()])))

        for box in bboxes:
            flat_box = []
            for elem in box:
                if isinstance(elem, list):
                    for item in elem:
                        flat_box.append(item)
                else:
                    flat_box.append(elem)
            new_bboxes.append(flat_box)

    return new_bboxes

def yolo_out(outs, anchors, image_h, image_w, input_shape, score_thres, nms_thres, keep_aspect_ratio, is_v5, sigmoid_in_post_v5, stride):
    num_layers = len(outs)
    masks = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    boxes, classes, scores = [], [], []

    if keep_aspect_ratio:
        if (input_shape[1] / image_h < input_shape[0] / image_w):
            input_shape_post = (input_shape[1], input_shape[1])
            h = image_h
            w = image_h
        else:
            input_shape_post = (input_shape[0], input_shape[0])
            h = image_w
            w = image_w
    else:
        h = image_h
        w = image_w
        input_shape_post = input_shape

    i = 0
    for out, mask in zip(outs, masks):
        if is_v5:
            if sigmoid_in_post_v5:
                b, c, s = process_feats_v5_sigmoid(out, input_shape_post, anchors, mask, stride[i])# v5: sigmoid in post
            else:
                b, c, s = process_feats_v5(out, input_shape_post, anchors, mask, stride[i])  # v5: sigmoid in model inference
        else:
            b, c, s = process_feats_v3(out, input_shape_post, anchors, mask, stride[i])#v3
        b, c, s = filter_boxes(b, c, s, score_thres)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
        i += 1

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # Scale boxes back to original image shape.
    image_dims = [w, h, w, h]
    boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, nms_thres)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

    if boxes.size > 0:
        boxes[:,:2] = np.rint(np.maximum(boxes[:,:2], 0))
        boxes[:,2:4] = np.rint(np.minimum(boxes[:,2:4], np.array([[image_w, image_h]])))

    return boxes, scores, classes

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_feats_v3(out, input_shape, anchors, mask, stride):
    grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

    anchors = [anchors[i] for i in mask]
    anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

    # Reshape to batch, height, width, num_anchors, box_params.
    out = out[0]
    box_xy = sigmoid(out[..., :2])
    box_wh = np.exp(out[..., 2:4])
    box_wh = box_wh * anchors_tensor

    box_confidence = sigmoid(out[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = sigmoid(out[..., 5:])

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= input_shape
    box_xy -= (box_wh / 2.)
    boxes = np.concatenate((box_xy, box_wh), axis=-1)

    return boxes, box_confidence, box_class_probs

def process_feats_v5(out, input_shape, anchors, mask, stride):
    grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

    anchors = [anchors[i] for i in mask]
    anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

    # Reshape to batch, height, width, num_anchors, box_params.
    out = out[0]
    box_xy = out[..., :2]
    box_wh = out[..., 2:4]
    box_wh = (box_wh * 2)*(box_wh * 2)*anchors_tensor

    box_confidence = out[..., 4]
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = out[..., 5:]

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy = box_xy*2-0.5+grid
    box_xy /= (grid_w, grid_h)
    box_wh /= input_shape
    box_xy -= (box_wh / 2.)
    boxes = np.concatenate((box_xy, box_wh), axis=-1)

    return boxes, box_confidence, box_class_probs

def process_feats_v5_sigmoid(out, input_shape, anchors, mask, stride):
    grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

    anchors = [anchors[i] for i in mask]
    anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

    # Reshape to batch, height, width, num_anchors, box_params.
    out = out[0]
    box_xy = sigmoid(out[..., :2])
    box_wh = sigmoid(out[..., 2:4])
    box_wh = (box_wh * 2)*(box_wh * 2)*anchors_tensor

    box_confidence = sigmoid(out[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = sigmoid(out[..., 5:])

    # grid_h * grid_w
    col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
    # grid_h * grid_w
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy = (box_xy * 2 - 0.5 + grid) * stride
    box_xy /= input_shape
    box_wh /= input_shape
    box_xy -= (box_wh / 2.)
    boxes = np.concatenate((box_xy, box_wh), axis=-1)

    return boxes, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs, score_thres):
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= score_thres)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores, nms_thres):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thres)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep
