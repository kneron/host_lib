import numpy as np
from numpy.lib.stride_tricks import as_strided

def _realnms(dets, only_max=False, iou_thres=0.35):

    """
    non-maximum suppression: if only_max, will ignore iou_thres and return largest score bbox.
    dets: list[list[x, y, w, h]]
    only_max: bool
    iou_thres: float between (0,1)
    """
    dets = np.array(dets)

    if len(dets) == 0:
        return []
    scores = dets[:, 4]
    order = np.argsort(scores)[::-1]
    dets = dets[order, :]
    if only_max:
        return np.array([dets[0]])
    
    x1, y1, w, h, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    x2, y2 = x1 + w - 1, y1 + h - 1

    areas = w * h

    order = scores.argsort()[::-1]

    keep_real = []
    tol = 0.1

    while order.size > 0:
        i = order[0]
        keep_real.append(i)
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

    return dets[keep_real,:]


def _pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)

def _nms(heat,hmax=None, kernel=3):
    # hmax = MaxPool2D(kernel, strides=1,padding='same')(heat)
    # heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    if hmax is None:
        hmax = np.zeros_like(heat)
        b, h, w, c = heat.shape
        for batch in range(b):
            for channel in range(c):
                hmax[batch,:,:,channel] = _pool2d(heat[batch,:,:,channel], kernel_size=kernel, stride=1, padding=1)
    assert (heat.shape == hmax.shape)
    # make non local max item zero
    keep = heat==hmax
    # heat[heat!=hmax] = 0
    return heat*keep

def postprocess_(outputs, max_objects=100, score_thres=0.5,
                 scale=None, input_shape=None, w_ori=None, h_ori=None,
                 nms=True, iou_thres=0.35, mapping_func='linear', **kwargs):

    assert len(outputs) % 3 == 0
    n_stage = len(outputs)//3
    dets = []
    batch_index = 0
    for stage in range(n_stage):

        """get output tensor and anchor tensor"""
        reg, cls, cts = outputs[stage], outputs[stage+n_stage], outputs[stage+n_stage*2]

        """get dimension info"""
        b, nrows, ncols, nclasses = cls.shape
        assert b==1
        # calculate here or pass by parameter
        stride = 2**int(np.log2(1.0*input_shape[0]/nrows)+0.5)

        """iterate over all anchors and select those with valid score"""
        # batch is always 0 here
        for i in range(nrows):
            for j in range(ncols):
                """class_id is the indice of labels. 0 is not background"""
                class_id = np.argmax(cls[batch_index, i, j, :])
                score = np.sqrt(cls[batch_index, i, j, class_id] * cts[batch_index, i, j, 0])

                if score > score_thres:
                    if mapping_func == 'exp':
                        l, t, r, b = np.exp(reg[0, i, j])
                    elif mapping_func == 'linear':
                        reg_relu = np.clip(reg[0, i, j], 0, 1e8)
                        l, t, r, b = (2**(3+stage)) * (reg_relu**2)
                    else:
                        assert 0
                    cx, cy = j*stride + stride//2, i*stride + stride//2
                    xmin, ymin, xmax, ymax = cx-l, cy-t, cx+r, cy+b

                    dets.append([xmin, ymin, xmax, ymax, score, class_id])

    dets = np.asarray(dets)
    if scale is not None:
        dets[..., :4] = dets[..., :4]*scale

    if w_ori is not None and h_ori is not None and np.size(dets)>0:
        # clip bbox make it inside image
        dets[..., :4] = np.clip(dets[..., :4], [0., 0., 0., 0.],
                                               np.c_[w_ori, h_ori, w_ori, h_ori])


    if len(dets) > 0:
        dets = np.asarray(dets)
        dets[..., 2:4] = dets[..., 2:4] - dets[..., :2]
        if nms:
            dets_real = _realnms(dets, only_max=max_objects == 1, iou_thres=iou_thres)
        else:
            dets_real = dets
    else:
        dets_real= []
    # [[x1,y1,w,h],[x1,y1,w,h]]
    dets_real = np.asarray(dets_real)
    dets_real = dets_real[..., :]
    return dets_real.tolist()
