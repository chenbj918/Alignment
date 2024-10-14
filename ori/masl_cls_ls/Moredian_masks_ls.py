import os
import cv2
import onnxruntime
import numpy as np

# ---------------------------------------landmark-----------------------------------------------
model_landmark = './landmaksPredTrain_no_module_154.onnx'

onnx_session = onnxruntime.InferenceSession(model_landmark)
input_name = []
for node in onnx_session.get_inputs():
    input_name.append(node.name)
output_name = []
for node in onnx_session.get_outputs():
    output_name.append(node.name)
print('input_name ', input_name)
print('output_name ', output_name)


def pre_processing(img_roi):
    img = cv2.resize(img_roi,(64,64))
    img = np.float32(img) / 255.0
    '''
    # norm
    means = np.array([0.5, 0.5, 0.5])
    stds  = np.array([0.5, 0.5, 0.5])
    img  -= means
    img  /= stds
    '''
    # channel first
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    # add batch axis
    img = img[np.newaxis, ...]

    return img


def Inference(img_roi):
    img_input = pre_processing(img_roi)

    input_feed = {}
    input_feed['input0'] = img_input
    output1,output2 = onnx_session.run(output_name, input_feed)

    return output1,output2


def after_processing(img_roi, begin_x, begin_y, result):
    h = img_roi.shape[0]
    w = img_roi.shape[1]

    result = np.squeeze(result)

    for n in range(0, 10):
        if n in [0,2,4,6,8]:
            result[n] = result[n] * (w/64) + begin_x
        else:
            result[n] = result[n] * (h/64) + begin_y

    return result


# ---------------------------------------masks-------------------------------------------------
model_masks = './mask_celebA+device.onnx'

onnx_session_mask = onnxruntime.InferenceSession(model_masks)
input_name_mask = []
for node in onnx_session_mask.get_inputs():
    input_name_mask.append(node.name)
output_name_mask = []
for node in onnx_session_mask.get_outputs():
    output_name_mask.append(node.name)
print('input_name_mask ', input_name_mask)
print('output_name_mask ', output_name_mask)


def pre_processing_mask(img_roi):
    img = cv2.resize(img_roi,(64,64))
    img = np.float32(img) / 255.0
    '''
    # norm
    means = np.array([0.5, 0.5, 0.5])
    stds  = np.array([0.5, 0.5, 0.5])
    img  -= means
    img  /= stds
    '''
    # channel first
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    # add batch axis
    img = img[np.newaxis, ...]

    return img


def Inference_mask(img_roi):
    img_rgb = img_roi[:, :, [2, 1, 0]]
    img_input = pre_processing_mask(img_rgb)

    input_feed = {}
    input_feed['input'] = img_input
    output = onnx_session_mask.run(output_name_mask, input_feed)

    return output


def after_processing_mask(output):
    x = np.array(output)
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum(axis=0)).tolist()


# 最终的人脸对齐图像尺寸分为两种：112x96和112x112，并分别对应结果图像中的两组仿射变换目标点,如下所示
imgSize1 = [112, 96]
imgSize2 = [112, 112]
imgSize3 = [128, 128]
imgSize4 = [96, 112]
coord5point1 = [[30.2946, 51.6963],  # 112x96的目标点
                [65.5318, 51.6963],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.3655]]
coord5point2 = [[30.2946 + 8.0000, 51.6963],  # 112x112的目标点
                [65.5318 + 8.0000, 51.6963],
                [48.0252 + 8.0000, 71.7366],
                [33.5493 + 8.0000, 92.3655],
                [62.7299 + 8.0000, 92.3655]]

coord5point3 = [[ 43.76525714,  59.08148571],  #128 * 128
                [ 84.03634286,  59.08148571],
                [ 64.0288    ,  79.98468571],
                [ 47.48491429, 102.56057143],
                [ 80.83417143, 102.56057143]]

coord5point4 = [[ 30.2976, 48.6992],  #96 * 112
                [ 65.5296, 48.4976],
                [ 48.0288, 68.7360],
                [ 33.5520, 88.3664],
                [ 62.7264, 88.2096]]

coord5point5 = [[30.2946, 44.3111],  # 96x96的目标点
                [65.5318, 44.3111],
                [48.0252, 61.4885],
                [33.5493, 79.1704],
                [62.7299, 79.1704]]


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


def warp_im(img_im, orgi_landmarks, tar_landmarks):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (96, 96))
    #dst = cv2.warpAffine(img_im, M[:2], (112, 112))
    return dst


def file_search(path, file_list, file_suffix = ['jpg', 'png', 'bmp']):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            file_search(filepath, file_list, file_suffix)
        else:
            if filepath.split(".")[-1] in file_suffix:
                file_list.append(filepath)
    return file_list


if __name__ == '__main__':

    # test img folder
    img_src = r'./test_img/img2'
    img_dst = r'./test_img'
    
    # show landmark
    Flag_show_landmark = False
    Flag_show_align = False
    Flag_show_mask = True

    # test img list
    img_list = []
    file_search(img_src, img_list)

    for idx, img_file in enumerate(img_list):
            print(img_file)
            # load img
            image = cv2.imread(img_file)
            if image is None:
                continue

            # parse det box
            box_info = os.path.basename(img_file)[:-4].split('_')
            x = int(box_info[-4][1:]) if 'x' in box_info[-4] else int(box_info[-4]) 
            y = int(box_info[-3][1:]) if 'y' in box_info[-3] else int(box_info[-3])
            w = int(box_info[-2][1:]) if 'w' in box_info[-2] else int(box_info[-2])
            h = int(box_info[-1][1:]) if 'h' in box_info[-1] else int(box_info[-1])
            rect = [x, y, w, h]

            # crop roi
            img_roi = image[y:y+h, x:x+w]
            
            # pre_processing
            pre_result1,pre_result2 = Inference(img_roi)

            # after_processing
            lmks = after_processing(img_roi, x, y, pre_result1)

            # show landmark
            if Flag_show_landmark:
                cv2.circle(image, (int(lmks[0]),int(lmks[1])), 1, (255, 0, 0), 2)
                cv2.circle(image, (int(lmks[2]),int(lmks[3])), 1, (255, 0, 0), 2)
                cv2.circle(image, (int(lmks[4]),int(lmks[5])), 1, (255, 0, 0), 2)
                cv2.circle(image, (int(lmks[6]),int(lmks[7])), 1, (255, 0, 0), 2)
                cv2.circle(image, (int(lmks[8]),int(lmks[9])), 1, (255, 0, 0), 2)

                cv2.imshow("landmark", image)
                cv2.waitKey(0)

            # align face
            lmks = np.array(lmks, dtype=np.float32)
            lmks = lmks.reshape((5, 2))
            img_align = warp_im(image, lmks, coord5point5)

            # show face align
            if Flag_show_align:
                cv2.imshow('img_align', img_align)
                cv2.waitKey(0)
            
            # Inference_mask
            raw_output = Inference_mask(img_align)
            score = after_processing_mask(raw_output)
            print(score)
            print(score.index(max(score)))
            
            # show mask
            if Flag_show_mask:
                cv2.putText(image, str('%.2f'%(score[0])), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0))
                cv2.putText(image, str('%.2f'%(score[1])), (50,110), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0))
                cv2.putText(image, str('%.2f'%(score[2])), (50,170), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0))
                cv2.imshow('img_mask', image)
                cv2.waitKey(0)


