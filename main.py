import numpy as np
import cv2
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import FPS
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util



#Main class for Monitor Camera
class Monitor():
	def __init__(self, *args, **kargs):
		#Go back into folder

		#Initialize centroid tracker, and blank arrays for storage
		self.args = kargs
		self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
		self.trackers = []
		self.trackableObjects = {}

		#Initialize counting arrays
		self.totalFrames = 0
		self.totalLeft = 0
		self.totalRight = 0
		self.log = []

		#Load TensorFlow Stuff
		self.loadTensorModel()
		self.loadLabelMap()

		#Initialize Threads for Reading Images and Showing
		self.vs = thread.VideoGet(self.args['input']).start()
		self.disp = thread.VideoShow(self.vs.frame).start()

		#Get Initial time
		self.now = datetime.datetime.now()

		self._counter = 1

		

	def loadTensorModel(self):
		self._MODEL_NAME = r'ssdlite_mobilenet_v2_coco_2018_05_09'
		self._MODEL_FILE = self._MODEL_NAME + r'.tar.gz'
		self._DOWNLOAD_BASE = r'http://download.tensorflow.org/models/object_detection/'

		self._PATH_TO_CKPT = self._MODEL_NAME + r'/frozen_inference_graph.pb'

		self._PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

		self._NUM_CLASSES = 90

		opener = urllib.request.URLopener()
		opener.retrieve(self._DOWNLOAD_BASE + self._MODEL_FILE, self._MODEL_FILE)
		tar_file = tarfile.open(self._MODEL_FILE)

		for file in tar_file.getmembers():
			file_name = os.path.basename(file.name)

			if r'frozen_inference_graph.pb' in file_name:
				tar_file.extract(file, os.getcwd())

	def loadLabelMap(self):
		self.detection_graph = tf.Graph()

		with self.detection_graph.as_default():
			self._od_graph_def = tf.compat.v1.GraphDef()

			with tf.compat.v2.io.gfile.GFile(self._PATH_TO_CKPT, 'rb') as fid:
				self._serialized_graph = fid.read()
				self._od_graph_def.ParseFromString(self._serialized_graph)
				tf.import_graph_def(self._od_graph_def, name='')

		self.label_map = label_map_util.load_labelmap(self._PATH_TO_LABELS)
		self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self._NUM_CLASSES, use_display_name=True)
		self.category_index = label_map_util.create_category_index(self.categories)

	def getCords(self, box):
		return (int(self.frame.shape[1]*box[1]), int(self.frame.shape[0]*box[0]), int(self.frame.shape[1]*box[3]), int(self.frame.shape[0]*box[2]))

	def run(self):
		'''
		Main function to run the object identification
		'''
		try:
			with self.detection_graph.as_default():
				with tf.compat.v1.Session(graph=self.detection_graph) as sess:
					#Start Counting FPS
					self.FPS = FPS().start()

					while True:
						#Check if User Asked to stop
						if self.disp.stopped or self.vs.stopped:
							self.vs.stop()
							break

						#Read Image
						self.frame = self.vs.read()
						rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

						#Setup tensorflow data
						image_np_expanded = np.expand_dims(self.frame, axis=0)
						image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

						(H, W) = self.frame.shape[:2]

						status = 'Waiting'
						rects = []

						if self.totalFrames % self.args['skip_frames'] == 0:
							#Activates every n frames to run computationally expensive dnn.
							status = 'Detecting'
							trackers = []

							boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
							# Cada pontuação representa o nível de confiança de cada um dos objetos.
							# A pontuação é mostrada na imagem do resultado, junto com o rótulo da classe.
							scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
							classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
							num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

							(boxes, scores, classes, num_detections) = sess.run(
								[boxes, scores, classes, num_detections],
								feed_dict={image_tensor: image_np_expanded}
								)


							boxes = np.squeeze(boxes)
							scores = np.squeeze(scores)
							classes = np.squeeze(classes)

							for i in range(boxes.shape[0]):
								if np.sum(boxes[i]) == 0 or scores[i] < self.args['confidence'] or classes[i] not in self.args['classes']:
									continue

								print('\n\nSUM\n')
								print(np.sum(boxes[i]))

								cords = boxes[i]
								confidence = scores[i]
								obj_type = classes[i]

								tracker = dlib.correlation_tracker()
								(startX, startY, endX, endY) = self.getCords(cords)

								rect = dlib.rectangle(startX, startY, endX, endY)

								tracker.start_track(rgb, rect)

								trackers.append([tracker, self.category_index[obj_type]['name']])



						else: #If we are not running dnn, run this
							for tracker in trackers:
								#Loop through all trackers in our previously defined list

								#Set status to display on screen.
								status = "Tracking"

								#Update position of each tracker
								tracker[0].update(rgb)
								pos = tracker[0].get_position()

								#Get coordinates of tracker box
								startX = int(pos.left())
								startY = int(pos.top())
								endX = int(pos.right())
								endY = int(pos.bottom())

								cv2.rectangle(self.frame, (startX,startY), (endX,endY), (255,0,0), 2)

								rects.append([(startX, startY, endX, endY),tracker[1]])

						#Draw Center line for visual indication of crossing
						cv2.line(self.frame, (W//2, 0), (W//2, H), (0,0,0), 3)

						#Update list of objects with new detection boxes
						self.objects = self.ct.update(rects)

						for (objectID, data) in self.objects.items():
							(centroid,category) = data

							#If we already initialized this item in the video, get its TrackableObject.
							to = self.trackableObjects.get(objectID, None)

							#If not, initiate a TrackableObject for it depending on where it started in the screen.
							if to is None:
								if centroid[0] < W // 2:
									to = TrackableObject(objectID, centroid, "Left", category)
								else:
									to = TrackableObject(objectID, centroid, "Right", category)

							else:
								to.centroids.append(centroid)

								#If we haven't counted the object, check conditions below.
								if not to.counted:
									if centroid[0] < W // 2 and to.start == "Right": #If object started on right and is now on left, count it with mark "Left"
										#Add count to our total, enter ObjectID into log, and mark it so we don't count again.
										self.totalLeft += 1
										self.log.append([objectID, datetime.datetime.now(), 'Left', category])
										to.counted = True
									elif centroid[0] > W // 2 and to.start == "Left": #If object started on left and is now on right, count it with mark "Right"
										#Add count to our total, enter ObjectID into log, and mark it so we don't count again.
										self.totalRight +=1
										self.log.append([objectID, datetime.datetime.now(), 'Right', category])
										to.counted = True

							#Store the trackable object in our dictionary
							self.trackableObjects[objectID] = to

							# draw both the ID of the object and the centroid of the
							# object on the output frame
							text = "ID {}".format(objectID)
							cv2.putText(self.frame, text, (centroid[0] - 10, centroid[1] - 10),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
							cv2.putText(self.frame, category.upper(), (centroid[0] - 10, centroid[1] + 20),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
							cv2.circle(self.frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)


						info = [
						("Right", self.totalRight),
						("Left", self.totalLeft),
						("Status", status),
						]

						#Label all objects we detect in frame
						for (i, (k, v)) in enumerate(info):
							text = f"{k}: {v}"
							cv2.putText(self.frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


						self.updateTime()
						self.FPS.update()
						print(self._counter)
						self.totalFrames += 1
						self._counter += 1
						self.disp.frame = self.frame


		finally:
			#Stop FPS
			self.FPS.stop()

			#Print FPS info to console
			print("[INFO] elapsed time: {:.2f}".format(self.FPS.elapsed()))
			print("[INFO] approx. FPS: {:.2f}".format(self.FPS.fps()))

			#Close all windows
			cv2.destroyAllWindows()

			self.close()


		return self

	def updateTime(self, forceLog = False):
		'''
		Determine whether time to log.
		'''
		if datetime.datetime.now().hour > self.now.hour or forceLog: #If new hour started, run this

			#Prepare filename
			filename = r'logs\\' + f'{self.now.year:04d}{self.now.month:02d}{self.now.day:02d}{self.now.hour:02d}00.csv'

			#Log files
			self.saveLogs(filename)

			#Reset all counters to begin anew
			self.now = datetime.datetime.now()
			self.totalLeft = 0
			self.totalRight = 0
			self.log = []
			self.ct.nextObjectID = 0

	def saveLogs(self, filename):
		'''
		Saves all log info into csv file.
		'''

		#If no logs folder exists, make one.
		if not os.path.isdir(os.path.join(os.path.dirname(__file__),r'logs')):
			os.mkdir(os.path.join(os.path.dirname(__file__),r'logs'))

		#Write all data, line by line, with index, time of crossing, and direction they went
		with open(filename, 'w') as file:
			writer = csv.writer(file, delimiter=',')
			for idx,date,direction,category in self.log:
				writer.writerow([idx, date.strftime("%d/%m/%Y %H:%M:%S"), category, direction])

	def close(self):
		self.vs.close()


if __name__ == '__main__':
	#Add argparser for console boot with input
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", default=os.path.join(os.path.dirname(__file__),r'mobilenet_sdd\MobileNetSSD_deploy.prototxt.txt'),
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", default=os.path.join(os.path.dirname(__file__),r'mobilenet_sdd\MobileNetSSD_deploy.caffemodel'),
		help="path to Caffe pre-trained model")
	ap.add_argument("-C", "--classes", default=[1, 2, 3, 4, 6, 17, 18],
		help="list of classes to be identified", type=list)
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	ap.add_argument("-c", "--confidence", type=float, default=0.50,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip fraqmes between detections")
	args = vars(ap.parse_args())

	#Close all windows before starting
	cv2.destroyAllWindows()

	#Start our main cam object and run it
	cam = Monitor(**args).run()

	#Close cam as last thing.
	cam.close()

	'''
	[1, 2, 3, 4, 6, 17, 18]
	'''