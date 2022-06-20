import torch
import cv2
import os
import uuid
import numpy as np
from centroidtracker import CentroidTracker
import time
from threading import Thread

print('start load model!!!')
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
model.conf = 0.5
model.iou = 0.4
print('load yolov5 successfully!!!')

if os.path.isdir('dataset') == False:
    os.mkdir('dataset')
tracker = CentroidTracker(maxDisappeared=10, maxDistance=110)
def cap_main(rtsp,num):
    cap = cv2.VideoCapture(rtsp)
    st = None
    while True:
        ret,frame = cap.read()
        frame = cv2.resize(frame, (640, 360))
        if ret == False:
            print(f'stop {rtsp}')
            break

        if st == None:
            st = time.time()
        et = time.time()
        if et - st > 0.2:

            results = model(frame, size=640)

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

                        person_img = frame[ymin:ymax,xmin:xmax]
                        filename = str(uuid.uuid4())
                        # cv2.imwrite(f'dataset/{filename}.jpg',person_img)
                        # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                        rects.append([xmin, ymin, xmax, ymax])

                boundingboxes = np.array(rects)
                rects = boundingboxes.astype(int)
                objects = tracker.update(rects)
                for (objectId, bbox) in objects.items():
                    x1, y1, x2, y2 = bbox
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    # frame_face = frame[y1:y2, x1:x2]
                    # gender, age = gender_age(frame_face,MODEL_MEAN_VALUES,ageList,genderList,faceNet,ageNet,genderNet)

                    objectId = objectId + 1
                    cv2.rectangle(frame, (x1 - 5, y1), (x2 - 5, y2), (0, 0, 255), 2)
                    text = "ID: {}".format(objectId)
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            cv2.imshow(f'{num}',frame)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

    cap.release()
    cv2.destroyWindow(f'{num}')

def cap_thread(rtsp,num):
    t = Thread(target=cap_main , args=(rtsp,num,))
    t.start()

if __name__ == '__main__':
    cap_thread('rtsp://test:advice128@110.49.125.237:554/cam/realmonitor?channel=1&subtype=0',1)
    cap_thread('rtsp://test:advice128@110.49.125.237:554/cam/realmonitor?channel=1&subtype=0',2)
    cap_thread('rtsp://test:advice128@110.49.125.237:554/cam/realmonitor?channel=1&subtype=0',3)
    # if num == 1:
    #     os.system('python main.py rtsp://admin:888888@192.168.1.50:10554/tcp/av0_0 1')
    # elif num == 2:
    #     os.system('python main.py rtsp://admin:888888@192.168.1.60:10554/tcp/av0_0 2')
    # elif num == 3:
    #     os.system('python main2.py rtsp_subtype 3')