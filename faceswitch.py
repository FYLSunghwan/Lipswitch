# -*- coding: utf-8 -*-
# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import tensorflow as tf
import numpy as np
 

# Variables declarations
frame_count=0
score=0
start = time.time()
pred=0
last=0
human_str=None
font=cv2.FONT_HERSHEY_TRIPLEX
font_color=(255,255,255)

modelFullPath = './tmp/output_graph.pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = './tmp/output_labels.txt'                                   # 읽어들일 labels 파일 경로

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

def create_graph():
	"""저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
	# 저장된(saved) graph_def.pb로부터 graph를 생성한다.
	with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

#Start tensorflow Session
create_graph()
with tf.Session() as sess:
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	while True:
		frame = vs.read()
		frame_count+=1
		if frame_count%5==0:
			# grab the frame from the threaded video stream, resize it to
			# have a maximum width of 400 pixels, and convert it to
			# grayscale
			frame = imutils.resize(frame, width=400)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

			# detect faces in the grayscale frame
			rects = detector(gray, 0)
	
			sw=0
			# loop over the face detections
			for rect in rects:
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# loop over the (x, y)-coordinates for the facial landmarks
				# and draw them on the image
				#for (x, y) in shape[48:68]:
					#cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
				(x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
				roi = frame[y-7:y + h+7, x-12:x + w + 12]
				roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
				frame=roi
			cv2.imwrite("current_frame.jpg",frame)
			answer=None
			if(len(rects)>0):
				image_data = tf.gfile.FastGFile("./current_frame.jpg", 'rb').read()
				predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})

				predictions = np.squeeze(predictions)
				top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
				f = open(labelsFullPath, 'rb')
				lines = f.readlines()
				labels = [str(w).replace("\n", "") for w in lines]
				answer = labels[top_k[0]]
				print last , " " , answer
			current= time.time()
			fps=frame_count/(current-start)

			# Show info during some time              
			if last<40 and frame_count>10:
				# Change the text position depending on your camera resolution
				cv2.putText(frame,answer, (20,400),font, 1, font_color)
				cv2.putText(frame,str(np.round(score,2))+"%",(20,440),font,1,font_color)

			if frame_count>20:
				fps_text="fps: "+str(np.round(fps,2))
				cv2.putText(frame, fps_text, (460,460), font, 1, font_color)
			cv2.imshow("Frame", frame)
			last+=1


			# if the 'q' key is pressed, stop the loop
			if cv2.waitKey(1) & 0xFF == ord("q"):break


# do a bit of cleanup

cv2.destroyAllWindows()
vs.stop()
sess.close()
print("Done")
