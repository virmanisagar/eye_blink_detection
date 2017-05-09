
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import random
import time
t_end = time.time() + 60 * 1


num = random.randrange(10,21,2)
print("No of times to blink - ", num+1)
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	return (A + B) / (2.0 * C)

def detect_blink(threshold=0.27, consec_frames=7):
	total, count = 0, 0

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('data.dat')
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
	#num = randrange(10,21,2)
	while time.time() < t_end:
		if total > num:
			print('Liveness detected: person is live')
			return True
		frame = vs.read()
		frame = imutils.resize(frame, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		for rect in rects:
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]
			leftEAR, rightEAR = eye_aspect_ratio(leftEye), eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			leftEyeHull, rightEyeHull = cv2.convexHull(leftEye), cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			if ear < threshold:
				count += 1
			else:
				if count >= consec_frames:
					total += 1
				count = 0
			cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	 		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	 	if key == ord("q"):
			break
	cv2.destroyAllWindows()
	vs.stop()

if __name__ == '__main__':
	detect_blink()
