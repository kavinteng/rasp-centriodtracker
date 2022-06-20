from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
import torch



#python main.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4

t0 = time.time()

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

def run(rtsp):
	print('start load model!!!')
	model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
	model.conf = 0.5
	model.iou = 0.4
	print('load yolov5 successfully!!!')

	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	# 	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	# 	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	# 	"sofa", "train", "tvmonitor"]
	#
	# # load our serialized model from disk
	# protopath = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
	# modelpath = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
	# net = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

	# if a video path was not supplied, grab a reference to the ip camera
	vs = cv2.VideoCapture(rtsp)

	# initialize the video writer (we'll instantiate later if need be)
	writer = None

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None

	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=10, maxDistance=90)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	totalout = 0
	totalin = 0
	x = []
	empty=[]
	empty1=[]

	# start the frames per second throughput estimator
	fps = FPS().start()
	#
	# if config.Thread:
	# 	vs = thread.ThreadingClass(config.url)

	# loop over frames from the video stream
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		ret,frame = vs.read()
		if ret == False:
			break

		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if frame is None:
			break

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % 10 == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
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
						# box = [xmin,ymin,xmax,ymax]
						# (startX, startY, endX, endY) = box.astype("int")


						# construct a dlib rectangle object from the bounding
						# box coordinates and then start the dlib correlation
						# tracker
						tracker = dlib.correlation_tracker()
						rect = dlib.rectangle(xmin,ymin,xmax,ymax)
						tracker.start_track(rgb, rect)

						# add the tracker to our list of trackers so we can
						# utilize it during skip frames
						trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 0), 3)
		cv2.line(frame, ((W // 2)-10, 0), ((W // 2)-10, H), (0, 0, 0), 3)
		cv2.line(frame, ((W // 2)+10, 0), ((W // 2)+10, H), (0, 0, 0), 3)
		cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		boundingboxes = np.array(rects)
		boundingboxes = boundingboxes.astype(int)
		rects = non_max_suppression_fast(boundingboxes, 0.3)
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[0] for c in to.centroids]

				direction = centroid[0] - np.mean(y)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not

				if not to.counted:
					# if the direction is negative (indicating the object
					# is moving up) AND the centroid is above the center
					# line, count the object
					if direction < 0 and ((W // 2)-10 < centroid[0] < W // 2):
						totalin += 1
						empty.append(totalin)
						to.counted = True

					# if the direction is positive (indicating the object
					# is moving down) AND the centroid is below the
					# center line, count the object
					# elif direction > 0 and ((W // 2)+10 > centroid[0] > W // 2):
					# 	totalout += 1
					# 	empty1.append(totalout)
					# 	#print(empty1[-1])
					# 	x = []
					# 	# compute the sum of total people inside
					# 	x.append(len(empty)-len(empty1))
					# 	#print("Total people inside:", x)
					# 	# if the people limit exceeds over threshold, send an email alert
					# 	# if sum(x) >= config.Threshold:
					# 	# 	cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
					# 	# 		cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
					# 	# 	if config.ALERT:
					# 	# 		print("[INFO] Sending email alert..")
					# 	# 		Mailer().send(config.MAIL)
					# 	# 		print("[INFO] Alert sent")
					#
					# 	to.counted = True


			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# construct a tuple of information we will be displaying on the
		info = [
		# ("Exit", totalout),
		("Enter", totalin),
		("Status", status),
		]

		info2 = [
		("Total people inside", x),
		]

		# Display the output
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# for (i, (k, v)) in enumerate(info2):
		# 	text = "{}: {}".format(k, v)
		# 	cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		# Initiate a simple log to save data at end of the day
		# if config.Log:
		# 	datetimee = [datetime.datetime.now()]
		# 	d = [datetimee, empty1, empty, x]
		# 	export_data = zip_longest(*d, fillvalue = '')
		#
		# 	with open('Log.csv', 'w', newline='') as myfile:
		# 		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		# 		wr.writerow(("End Time", "In", "Out", "Total Inside"))
		# 		wr.writerows(export_data)


		# show the output frame
		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

		# if config.Timer:
		# 	# Automatic timer to stop the live stream. Set to 8 hours (28800s).
		# 	t1 = time.time()
		# 	num_seconds=(t1-t0)
		# 	if num_seconds > 28800:
		# 		break

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


	# # if we are not using a video file, stop the camera video stream
	# if not args.get("input", False):
	# 	vs.stop()
	#
	# # otherwise, release the video file pointer
	# else:
	# 	vs.release()

	# close any open windows
	cv2.destroyAllWindows()


##learn more about different schedules here: https://pypi.org/project/schedule/
# if config.Scheduler:
# 	##Runs for every 1 second
# 	#schedule.every(1).seconds.do(run)
# 	##Runs at every day (9:00 am). You can change it.
# 	schedule.every().day.at("9:00").do(run)
#
# 	while 1:
# 		schedule.run_pending()
#
# else:
# run('rtsp://test:advice128@110.49.125.237:554/cam/realmonitor?channel=1&subtype=0')
run('rtsp://admin:888888@192.168.7.50:10554/tcp/av0_0')
