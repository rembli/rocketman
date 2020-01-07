from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn import cluster as cluster_models
from peakdetect import peakdetect
import os

#######################################################################
# SETTING
#######################################################################

data_root = "C:\Data\Dev-Data\music\\"
if os.getenv("ROCKETMAN_DATA") is not None:
    data_root = os.getenv("ROCKETMAN_DATA")

fpath_pieces = data_root + "pieces\\"
fpath_lines = data_root + "lines\\"
fpath_notes = data_root + "notes\\"
cpath_notes = data_root + "cluster\\"

target_height = 500
brightening_factor = 1
threshold = 0.7

similarity = 0.65

##################################################################################
# DEF: RECTANGLE
##################################################################################

class Rectangle:
    def __init__(self, min_x, min_y, max_x, max_y, identificator):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.identificator = identificator

    def __repr__(self):
        return str(self.identificator) + ": " + str(self.min_x) + "/" + str(self.min_y) + "/" + str(
            self.max_x) + "/" + str(self.max_y)

    def intersects(self, other):
        return not (self.max_x < other.min_x or
                    self.max_y < other.min_y or
                    self.min_x > other.max_x or
                    self.min_y > other.max_y)

    def vintersects(self, other):
        return (self.max_x > other.min_x and other.max_x > self.min_x)


for file in os.listdir(fpath_pieces):
    fname, ftype = file.split(".",2)
    ftype = "."+ftype
    filename = fpath_pieces + fname + ftype
    print ("***",filename,"***")

    ##################################################################################
    # IMAGE PREPROCESSING
    print ("IMAGE PREPROCESSING ---")
    ##################################################################################

    image = plt.imread(filename)

    # resize
    print ("--- original size:", image.shape)
    height, width, rgb = image.shape
    sizing_factor = height / target_height
    image = cv2.resize(image,(int(width/sizing_factor),target_height))
    print ("--- new size:",image.shape)

    # gray scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # brighten
    # matrix = np.ones(image.shape, dtype = "uint8") * brightening_factor
    # image = cv2.add(image, matrix, dtype = cv2.CV_8UC1)

    # black & white
    # val, image = cv2.threshold(image, 50, 255, cv2.THRESH_OTSU)

    ##################################################################################
    # PREPARE FOR DBSCAN CLUSTERING
    print ("PREPARE IMAGE FOR DBSCAN CLUSTERING OF IMAGE COMPONENTS")
    ##################################################################################

    # color image
    # height, width, rgb = image.shape

    # black/white image
    height, width = image.shape

    image_scatter = list()
    for y in range(height):
        for x in range(width):

            gray = image[y, x]
            if gray < threshold:
                image_scatter.append([x, y])

    image_scatter_array = np.array(image_scatter)



    ##################################################################################
    # CLUSTERING
    ##################################################################################

    print ("RUN CLUSTERING")
    model = cluster_models.DBSCAN(eps=10)
    clusters = model.fit_predict (image_scatter_array)

    ##################################################################################
    # COMBINE OVERLAPPING CLUSTER
    ##################################################################################

    print ("COMBINE OVERLAPPING CLUSTER")
    rects = list()

    # STEP 1: für jedes identifizierte cluster alle Positionen in ein eigenes array übertragen
    y = np.array(clusters)
    num_cluster = y.max()

    cluster_image_array = list()
    for i in range(num_cluster):
        c = np.array([image_scatter_array[x] for x in range(1, y.shape[0]) if y[x] == i])

        # Sonderbehandlung für Aussreisser
        min_x = c[:, 1].min()
        max_x = c[:, 1].max()
        min_y = c[:, 0].min()
        max_y = c[:, 0].max()
        if not (max_x - min_x) > (height - 50):
            cluster_image_array.append(c)

    # STEP 2: aus dem array für das Cluster die min und max Werte der X und Y-Position ermitteln
    for i in range(len(cluster_image_array)):
        cluster_image = cluster_image_array[i]

        min_x = cluster_image[:, 1].min()
        min_y = cluster_image[:, 0].min()
        max_x = cluster_image[:, 1].max()
        max_y = cluster_image[:, 0].max()

        r = Rectangle(min_x, min_y, max_x, max_y, i)
        rects.append(r)

    # STEP 3: Cluster zusammen fassen
    cnt = 0
    num_rects = len(rects)
    while cnt < len(rects):
        rect = rects[cnt]
        # print (str(cnt)+": "+str(rect.identificator))
        for cnt2 in range(cnt + 1, len(rects)):
            rect2 = rects[cnt2]
            if rect.intersects(rect2):
                num_rects = num_rects + 1
                print ("--- cluster ", rect.identificator, " collides with ", rect2.identificator)
                # print ("----- add: ", num_rects)
                # print ("----- remove", rect.identificator)
                # print ("----- remove", rect2.identificator)

                ci1 = cluster_image_array[cnt]
                ci2 = cluster_image_array[cnt2]
                ci = np.concatenate((ci1, ci2))
                cluster_image_array.append(ci)

                min_y = ci[:, 0].min()
                max_y = ci[:, 0].max()
                min_x = ci[:, 1].min()
                max_x = ci[:, 1].max()
                r = Rectangle(min_x, min_y, max_x, max_y, num_rects)
                rects.append(r)

                del (cluster_image_array[cnt])
                del (cluster_image_array[cnt2 - 1])

                del (rects[cnt])
                del (rects[cnt2 - 1])

                cnt = -1
                break
        cnt = cnt + 1

    ##################################################################################
    # CAPTURE SINGLE LINE
    print ("EXTRACT SINGLE LINE OF NOTES")
    ##################################################################################

    min_height = 30
    min_width  = 300

    selected_images = []

    cnt = 0
    for i in range (len(cluster_image_array)):
        cluster_image = cluster_image_array[i]
        min_y= cluster_image[:,0].min()
        max_y= cluster_image[:,0].max()
        min_x= cluster_image[:,1].min()
        max_x= cluster_image[:,1].max()
        if max_y > min_y + min_width and max_x > min_x + min_height and max_y- min_y> max_x- min_x:
            selected_images.append (image[min_x:max_x,min_y:max_y])
            plt.figure(figsize = (20,10))
            plt.imshow(image[min_x:max_x,min_y:max_y], cmap="gray")
            plt.title ("Cluster {}".format(rects[i].identificator))
            plt.savefig (fpath_lines+fname+"_"+str(cnt)+ftype)
            cnt = cnt + 1

    print ("---", len(selected_images),"lines extracted")

    ##################################################################################
    # CAPTURE SINGLE NOTE
    print ("EXTRACT SINGLE NOTES")
    ##################################################################################

    crop_images = []

    for cnt in range(0, len(selected_images)):

        img = selected_images[cnt]
        img_height = img.shape[0]
        img_width = img.shape[1]

        # calcuate density
        x = 0
        density_array = []
        while x < img_width:
            density = sum(255 - img[:, x])
            density_array.append(density)
            x = x + 1

        # rescale density to value 0 - 1
        max_val = max(density_array)
        density_array = (density_array / max_val)

        # detect lows
        peaks = peakdetect(density_array, lookahead=15)

        lows = np.asarray(peaks[1])
        xl = lows[:, 0]
        yl = lows[:, 1]

        highs = np.asarray(peaks[0])
        xh = highs[:, 0]
        yh = highs[:, 1]

        # crop single notes
        i = 0
        x1 = 0
        for low in lows:
            x2 = int(low[0])
            crop_img = img[0:img_height, x1:x2] * 255

            # having the same image shape will improve performance
            smaller_than_50 = 50 - (x2 - x1)
            if (smaller_than_50) > 0:
                empty_array = np.full((img_height, smaller_than_50), 255)
                crop_img = np.concatenate((crop_img, empty_array), axis=1)
            crop_images.append(crop_img)

            # convert to black and white
            crop_img = crop_img.astype(np.uint8)
            retval, crop_img = cv2.threshold(crop_img, 50, 255, cv2.THRESH_OTSU)

            # save image to disk
            tmp = Image.fromarray(crop_img.astype(np.uint32))
            tmp_name = fpath_notes + fname + "_" + str(cnt) + "_" + str(i) + ftype
            tmp.convert("L").save(tmp_name)
            print ("--- saved",tmp_name)

            x1 = x2
            i = i + 1

print ("DONE!")

