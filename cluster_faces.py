# import the necessary packages
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
import os
import shutil

directory = "output"
parent_dir = os.getcwd() 
path = os.path.join(parent_dir, directory)

try:
	shutil.rmtree(path)
except:
	pass

try: 
	os.mkdir(path) 
except OSError as error:
	print(error)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of parallel jobs to run (-1 will use all CPUs)")
args = vars(ap.parse_args())
# load the serialized face encodings + bounding box locations from
# disk, then extract the set of encodings to so we can cluster on
# them
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]
imgPath = [d["imagePath"] for d in data]
# cluster the embeddings
print("[INFO] clustering...")
clt = DBSCAN(metric="euclidean", n_jobs=args["jobs"],min_samples=1)
clt.fit(encodings)	
# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

for j in labelIDs:
	directory2 = f"person{j}"
	parent_dir2 = os.getcwd()
	path2 = os.path.join(parent_dir2, "output") 
	path3 = os.path.join(path2, directory2)
	try:
		shutil.rmtree(path3)
	except:
		pass
	try: 
		os.mkdir(path3) 
	except OSError as error:
		print(error)

# loop over the unique face integers
for labelID in labelIDs:
	# find all indexes into the `data` array that belong to the
	# current label ID, then randomly sample a maximum of 25 indexes
	# from the set
	print("[INFO] faces for face ID: {}".format(labelID))
	idxs = np.where(clt.labels_ == labelID)[0]
	for i in idxs:

		directory2 = f"person{labelID}"
		parent_dir2 = os.getcwd()
		path2 = os.path.join(parent_dir2, "output") 
		path3 = os.path.join(path2, directory2)
		path4 = os.path.join(path3, f"pic{i}.jpg")

		src_direc = data[i]["imagePath"]
		path2 = os.path.join(parent_dir2,src_direc)
		src_path = path2
		dst_path = path4
		shutil.copy(src_path, dst_path)