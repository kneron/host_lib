import numpy as np

def softmax(logits):
    """
    softmax for logits like [[[x1,x2], [y1,y2], [z1,z2], ...]]
    minimum and maximum here work as preventing overflow
    """
    # print("logit", logits.shape)

    clas = np.exp(np.minimum(logits, 22.))
    clas = clas / np.maximum(np.sum(clas, axis=-1, keepdims=True), 1e-10)
    return clas

def nms(dets, only_max = False, iou_thres=0.35):
    """
    non-maximum suppression: if only_max, will ignore iou_thres and return largest score bbox.
    dets: list[list[x, y, w, h]]
    only_max: bool
    iou_thres: float between (0,1)
    """
    if len(dets) == 0:
        return []

    dets.sort(key = lambda x: x[4])
    if only_max:
        return [dets[-1]]

    dets = np.array(dets)
    x1, y1, w, h, scores = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
    x2, y2 = x1+w-1, y1+h-1

    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(dets[i,:])

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        inter_w = np.maximum(0.0, xx2 - xx1 + 1)
        inter_h = np.maximum(0.0, yy2 - yy1 + 1)
        
        inter_area = inter_w * inter_h
        iou = inter_area / (areas[i] + areas[order[1:]] - inter_area)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def postprocess_(raw_res, anchor_path, input_shape, w_ori, h_ori, score_thres, only_max, iou_thres):
    """
    decode one sample, among SSD output batch to rectangle.
    raw_res: list(np.ndarry), need reorganize to 4 X [h, w, anchors*num_class] + 4 X [h, w, anchors*4]
    anchor_path: path for the face anchors.
    w_ori: int, original image's width
    h_ori: int, original image's height
    score_thres: float
    only_max: bool, whether to do nms or just return one with largest score
    iou_thres: float, for nms
    nms: bool, whether to do nms
    """
    nstages = len(raw_res) // 2

    outputs = []
    # reorganize the raw data according to the requirement of postprocessing
    if (raw_res[0].shape[2] < raw_res[1].shape[2]):
        idx_box = len(raw_res) - 1
        idx_cls = len(raw_res) - 2
    else:
        idx_box = len(raw_res) - 2
        idx_cls = len(raw_res) - 1

    #class nodes
    for i in range(nstages):
        outputs.append(np.expand_dims(raw_res[idx_cls - 2*i], 0))

    # box nodes
    for i in range(nstages):
        outputs.append(np.expand_dims(raw_res[idx_box - 2*i], 0))

    # for raw_data in raw_res:
    #     print(raw_data.shape)

    # for raw_data in outputs:
    #     print(raw_data.shape)

    anchor = np.load(anchor_path, encoding="latin1", allow_pickle=True)
    anchor = anchor.tolist()

    w = input_shape[1]
    h = input_shape[0]

    scale = max(1.0 * w_ori / w, 1.0 * h_ori / h)

    dets = []
    nstages = len(outputs) // 2
    for stage in range(nstages):

        """get output tensor and anchor tensor"""
        logits = outputs[stage][0,...]
        regr = outputs[stage+nstages][0,...]
        anchor_box = anchor[str(stage)][0]

        """get dimension info"""
        nrows, ncols, nanchors = anchor_box.shape[0], anchor_box.shape[1], anchor_box.shape[2]
        # print(logits.shape, nanchors)
        nclasses = logits.shape[2] // nanchors

        """convert logits to probilities"""
        clas = softmax(logits.reshape(logits.shape[:-1]+(nanchors, nclasses)))

        # print(clas.shape)

        """iterate over all anchors and select those with valid score"""
        for i in range(nrows):
            for j in range(ncols):
                for k in range(nanchors):

                    """class_id is the indice of labels. 0 is background"""
                    class_id = np.argmax(clas[i, j, k, :])
                    if class_id == 0:
                        continue

                    score = clas[i, j, k, class_id]
                    if score > score_thres:
                        cx, cy = anchor_box[i,j,k,0]*input_shape[1]*scale, anchor_box[i,j,k,1]*input_shape[0]*scale
                        w, h = anchor_box[i,j,k,2]*input_shape[1]*scale, anchor_box[i,j,k,3]*input_shape[0]*scale

                        """add offset"""
                        cx, cy = cx + w*regr[i,j,4*k], cy + h*regr[i,j,4*k+1]
                        w, h = np.exp(regr[i,j,4*k+2])*w, np.exp(regr[i,j,4*k+3])*h

                        dets.append([cx-w//2, cy-h//2, w, h, score, class_id])
    # do nms
    dets = nms(dets, only_max, iou_thres)

    if len(dets) > 0:
        # make dets stay inside image
        dets = np.array(dets)
        dets[:,2:4] = dets[:,:2] + dets[:,2:4]
        dets[..., :4] = np.clip(dets[..., :4], [0., 0., 0., 0.],
                                np.c_[w_ori, h_ori, w_ori, h_ori])
        dets[:,2:4] = dets[:,2:4] - dets[:,:2]
    return dets
