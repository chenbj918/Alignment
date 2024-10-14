import os
import cv2
import onnxruntime
import numpy as np

# ---------------------------------------landmark-----------------------------------------------
class cbjLandMarksDetector:
    def __init__(self,model_path=None):
        self.model_path = model_path
        assert os.path.exists(self.model_path)
        self.onnx_session = onnxruntime.InferenceSession(self.model_path)
        self.input_name = []
        self.output_name = []
        self.input_mean = 0.0
        self.input_val = 255.0
        for node in self.onnx_session.get_inputs():
            self.input_name.append(node.name)
            self.input_shape = node.shape
        for node in self.onnx_session.get_outputs():
            self.output_name.append(node.name)

    def expand_border(self, img, box):
        h = box[3]-box[1]
        w = box[2]-box[0]
        x1 = max(0,box[0]-w/10)
        y1 = max(0,box[1]-h/10)
        x2 = min(box[2]+w/10,img.shape[1]-1)
        y2 = min(box[3]+h/10,img.shape[0]-1)	
        return x1,y1,x2,y2

    def make_border(self, img):
        h,w,_ = img.shape
        if w > h:
            border = (w-h)/2
            new_img = cv2.copyMakeBorder(img, int(border), int(border), 0, 0, cv2.BORDER_CONSTANT, 0)        
        elif w < h:
            border = (h-w)/2
            new_img = cv2.copyMakeBorder(img, 0, 0, int(border), int(border), cv2.BORDER_CONSTANT, 0)
        else:
            border = 0
            new_img = img  
        return new_img, border

    def pre_processing(self, img_path, box):
        img = cv2.imread(img_path)
        x1,y1,x2,y2 = self.expand_border(img, box) #扩边
        img = img[int(y1):int(y2),int(x1):int(x2),:] #抠图
        img, border = self.make_border(img) #补边

        w_ = int(x2-x1) 
        h_ = int(y2-y1)
        w = img.shape[1]
        h = img.shape[0]

        img = cv2.resize(img,(self.input_shape[2],self.input_shape[3]))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.transpose(img,(2,0,1))
        img = np.array(img[np.newaxis,...],dtype=np.float32)
        img = (img-self.input_mean)/self.input_val #归一化

        return img,w,h,border,w_,h_,x1,y1

    def after_processing(self, img_path, box, result):
        img,w,h,border,w_,h_,x1,y1 = self.pre_processing(img_path, box)

        result = np.squeeze(result)
        #resize
        for i in range(0, 10):
            if i in [0,2,4,6,8]:
                result[i]=(result[i]*w)/self.input_shape[3]
            else:
                result[i]=(result[i]*h)/self.input_shape[2]
        result = result.tolist()
        #抠图并补边
        if w_ >= h_:
            result = np.array(result)
            for i in range(0, 10):
                if i in [0,2,4,6,8]:
                    result[i]=result[i]
                else:
                    result[i]=result[i]-border
            result = result.tolist()
        else:
            result = np.array(result)
            for i in range(0, 10):
                if i in [0,2,4,6,8]:
                    result[i]=result[i]-border
                else:
                    result[i]=result[i]
            result = result.tolist()
        #扩边
        result = np.array(result)
        for i in range(0, 10):
            if i in [0,2,4,6,8]:
                result[i]=(result[i]+x1)
            else:
                result[i]=(result[i]+y1)
        result = result.tolist()

        return result

    def __call__(self, img_path, box):
        img,w,h,border,w_,h_,x1,y1 = self.pre_processing(img_path, box)
        pre_result = self.onnx_session.run(self.output_name, {self.input_name[0] : img})
        after_result = self.after_processing(img_path, box, pre_result)
        return after_result


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
    U, S, Vt = np.linalg.svd(points1.T * points2) #svd奇异值分解
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


def warp_im(img_im, orgi_landmarks, tar_landmarks):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)  #计算仿射变换矩阵
    # dst = cv2.warpAffine(img_im, M[:2], (96, 96))
    dst = cv2.warpAffine(img_im, M[:2], (112, 112))
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
    img_src = 'F:/Face_Quality/blur/data/v1/1'

    # model
    model_landmark = 'F:/Alignment/landmaksPredTrain_no_module_383_0.07884941281022717.onnx'
    
    # show landmark
    Flag_show_landmark = True
    Flag_show_align = True

    # test img list
    img_list = []
    file_search(img_src, img_list)

    for idx, img_file in enumerate(img_list):
            print(img_file)
            # load img
            image = cv2.imread(img_file)
            if image is None:
                continue
            
            box = [0, 0, image.shape[1], image.shape[0]]

            lmks_detector = cbjLandMarksDetector(model_landmark)
            lmks = lmks_detector(img_file, box)

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
            img_align = warp_im(image, lmks, coord5point2)

            # show face align
            if Flag_show_align:
                cv2.imshow('img_align', img_align)
                cv2.waitKey(0)
            

