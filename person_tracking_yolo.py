import cv2
import numpy as np
from centroidtracker import CentroidTracker
import time
from threading import Thread
import sys
import torch

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def load_all_model():
    global MODEL_MEAN_VALUES,ageList,genderList,faceNet,ageNet,genderNet,model

    print('start load model!!!')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    model.conf = 0.5
    model.iou = 0.4
    print('load yolov5 successfully!!!')

    print('load gender & age model')
    faceProto="gender_age_model/opencv_face_detector.pbtxt"
    faceModel="gender_age_model/opencv_face_detector_uint8.pb"
    ageProto="gender_age_model/age_deploy.prototxt"
    ageModel="gender_age_model/age_net.caffemodel"
    genderProto="gender_age_model/gender_deploy.prototxt"
    genderModel="gender_age_model/gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)
    print('load gender & age successfully!!!')

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def gender_age(frame,faceNet,ageNet,genderNet):
    global gender,age
    padding = 20
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    # if not faceBoxes:
    #     print("No face detected")
    if len(faceBoxes) > 0:
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                        :min(faceBox[2] + padding, frame.shape[1] - 1)]
            if len(face) != 0:
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                # gender_array.append(gender)

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                # age_array.append(age)

                return gender,age
        return '-', '-'
    else:
        return '-','-'

def main(rtsp):
    # cap = cv2.VideoCapture("test_video.mp4")
    cap = cv2.VideoCapture(rtsp)
    st = None
    print(f'start cam: {rtsp}')
    while True:
        ret, frame = cap.read()
        if ret == False:
            print(f'stop {rtsp}')
            break
        if frame is None:
            continue
        if st == None:
            st = time.time()
        et = time.time()
        if et - st > 0.15:
            frame = cv2.resize(frame,(640,360))
            results = model(frame, size=320)
            out2 = results.pandas().xyxy[0]

            if len(out2) != 0:
                rects = []
                for i in range(len(out2)):
                    output_landmark = []
                    xmin = int(out2.iat[i, 0])
                    ymin = int(out2.iat[i, 1])
                    xmax = int(out2.iat[i, 2])
                    ymax = int(out2.iat[i, 3])
                    obj_name = out2.iat[i, 6]
                    if obj_name != 'person':
                        continue
                    if obj_name == 'person' or obj_name == '0':
                        rects.append([xmin,ymin,xmax,ymax])
            # tracking config & non max suppression
            # print(rects)
            boundingboxes = np.array(rects)
            boundingboxes = boundingboxes.astype(int)
            rects = non_max_suppression_fast(boundingboxes, 0.3)
            # print(rects)
            objects = tracker.update(rects)
            for (objectId, bbox) in objects.items():
                x1, y1, x2, y2 = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                frame_face = frame[y1:y2, x1:x2]
                gender,age = gender_age(frame_face, faceNet, ageNet, genderNet)

                objectId = objectId+1
                cv2.rectangle(frame, (x1-5, y1), (x2-5, y2), (0, 0, 255), 2)
                text = "ID: {} {} {}".format(objectId,gender,age)
                cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            st = time.time()
            cv2.imshow(f"{rtsp}", frame)

            k = cv2.waitKey(1)
            if k == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def main_threading(rtsp):
    m = Thread(target=main, args=(rtsp,))
    # m.daemon = True
    m.start()

if __name__ == '__main__':
    rtsp_input = sys.argv[1]
    if rtsp_input == 'rtsp_subtype':
        rtsp_input = 'rtsp://test:advice128@110.49.125.237:554/cam/realmonitor?channel=1&subtype=0'
    load_all_model()
    # load model
    # protopath = "camcount/MobileNetSSD_deploy.prototxt"
    # modelpath = "camcount/MobileNetSSD_deploy.caffemodel"
    # detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
    # load class in mobilenet
    # CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    #            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    #            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    #            "sofa", "train", "tvmonitor"]
    # get centroid tracker object
    tracker = CentroidTracker(maxDisappeared=10, maxDistance=90)
    main(rtsp_input)
    # main('rtsp://admin:888888@192.168.7.50:10554/tcp/av0_0')
    # main('rtsp://admin:888888@192.168.7.60:10554/tcp/av0_0')
    # main('rtsp://test:advice128@110.49.125.237:554/cam/realmonitor?channel=1&subtype=0')
    # 'rtsp://test:advice128@110.49.125.237:554/cam/realmonitor?channel=1&subtype=0'

    # main_threading('rtsp://admin:888888@192.168.7.50:10554/tcp/av0_0')
    # main_threading('rtsp://admin:888888@192.168.7.60:10554/tcp/av0_0')
    # main_threading('rtsp://test:advice#128@110.49.125.237:554/cam/realmonitor?channel=1&subtype=0')