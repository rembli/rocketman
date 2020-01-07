import sys
import pygame
import os

from PIL import Image
import matplotlib.pyplot as plt
import cv2

from soundshift import pitch
from scipy.io import wavfile

import numpy as np
import pandas as pd
from sklearn import cluster as cluster_models
from peakdetect import peakdetect
from keras.models import model_from_json

#######################################################################
# SETTING
#######################################################################

# FILE TO BE PLAYED
data_root = "C:\Data\Dev-Data\music\\"
if os.getenv("ROCKETMAN_DATA") is not None:
    data_root = os.getenv("ROCKETMAN_DATA")

piece = "10.png"
if (len (sys.argv))>1:
    piece = sys.argv[1]

filename = data_root+"pieces\\"+piece

# PATHS
model_path = data_root+"model\\"
sound_file = data_root+"sounds\\bowl.wav"
input_shape_images = (255,50,3)

# IMAGE SETTING
target_height = 500
brightening_factor = 1
threshold = 0.7

# DISPLAY SETTINGS
screen_caption = "scan 2 sound"
screen_bg_color = (255, 255, 255)
screen_width = 700
screen_height = 1100
screen_height_notes = 800

# INIT SOUNDMIXER (important: before paygame.init!)
sampleRate = 44100

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

##################################################################################
# PREPARE FOR DBSCAN CLUSTERING
print ("PREPARE IMAGE FOR DBSCAN CLUSTERING OF IMAGE COMPONENTS")
##################################################################################

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
eps = 10
model = cluster_models.DBSCAN(eps=eps)
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
            print("--- cluster ", rect.identificator, " collides with ", rect2.identificator)

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
print("EXTRACT SINGLE LINE OF NOTES")
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
        cnt = cnt + 1

print ("---", len(selected_images),"lines extracted")
if len(selected_images) <= 0:
    print ("ERROR: NO IMAGES COULD BE IDENTIFIED")
    sys.exit()

##################################################################################
# CAPTURE SINGLE NOTE
print("EXTRACT SINGLE NOTES")
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
        if(smaller_than_50) > 0:
            empty_array = np.full((img_height, smaller_than_50), 255)
            crop_img = np.concatenate((crop_img, empty_array), axis=1)

        # convert to black and white
        crop_img = crop_img.astype(np.uint8)
        retval, crop_img = cv2.threshold(crop_img, 50, 255, cv2.THRESH_OTSU)

        # append image
        crop_images.append(crop_img)

        x1 = x2
        i = i + 1

print ("--- extracted notes, let's play!")

#######################################################################
# LOAD MODEL
print ("LOAD MODEL")
#######################################################################

# load json and create model
json_file = open(model_path + 'model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# load weights into new model
model.load_weights(model_path + "model.h5")

# load model columns
train = pd.read_csv(model_path + "model-columns.csv")

print("--- Loaded model from disk")

#######################################################################
# PREDICT ON IMAGES AND PLAY
print ("PREDICT ON IMAGES")
#######################################################################

# INIT SCREEN
pygame.mixer.pre_init(sampleRate, -16, 1)
pygame.init()
pygame.display.set_caption(screen_caption)
screen = pygame.display.set_mode((screen_width, screen_height))

# LOAD SOUNDS
note_names = ["a", "g", "f", "e", "d", "c", "h", "a", "g", "f", "e", "d", "c", "h", "a", "g", "f", "e", "d", "c", "h",
              "a", "g", "f", "e", "d", "c", "h"]
num_notes = len(note_names)

fps, sound = wavfile.read(sound_file)
transposed_sounds = [pitch(sound, n) for n in range(num_notes, 0, -1)]
sounds = list(map(pygame.sndarray.make_sound, transposed_sounds))

# LOAD NOTES TO BE PLAYED
image = pygame.image.load(filename)
image_width, image_height = image.get_rect().size

image_scaling_factor_x = screen_width / image_width
image_scaling_factor_y = screen_height_notes / image_height
image_scaling_factor = image_scaling_factor_x if image_scaling_factor_x < image_scaling_factor_y else image_scaling_factor_y

image = pygame.transform.scale(image, (int(image_width * image_scaling_factor), int(image_height * image_scaling_factor)))
image_width, _ = image.get_rect().size

# FINALLY: PREDICT AND PLAY :-)
num_images = len (crop_images)
current_image = 0
is_playing = False

while True:
    #######################################################################
    # DRAW SCREEN
    #######################################################################

    img = crop_images[current_image]

    # convert to proper shape if loaded from mem (and not from disk)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((50 * 3, 255), Image.ANTIALIAS)
    img = np.array(pil_img)
    img = np.resize(img, (255, 50, 3))

    # reshaping extratec image for tensorflow
    imgT = img / 255
    imgT = imgT.reshape(1, 255, 50, 3)

    classes = np.array(train.columns[3:])
    proba = model.predict(imgT)
    top_3 = np.argsort(proba[0])[:-4:-1]

    # setup display
    screen.fill(screen_bg_color)
    screen.blit(image, (0, 0))

    line_color = (0, 0, 0)
    pygame.draw.line(screen, line_color, (10, screen_height_notes), (screen_width - 10, screen_height_notes))
    pygame.draw.line(screen, line_color, (screen_width/3, screen_height_notes+10), (screen_width/3, screen_height-10))
    pygame.draw.line(screen, line_color, (screen_width/3*2, screen_height_notes+10), (screen_width/3*2, screen_height-10))

    font = pygame.font.SysFont("comicsansms", 14, bold=True)
    font_color = (255, 255, 255)
    if is_playing:
        pygame.draw.rect(screen, (128,0,0), (10, screen_height_notes-25, screen_width-20, 20))
        text = font.render("Press space to STOP playing", True, font_color)
        screen.blit(text, (screen_width/3, screen_height_notes - 24))
    else:
        pygame.draw.rect(screen, (0, 128, 0), (10, screen_height_notes - 25, screen_width-20, 20))
        text = font.render("Press space to START playing", True, font_color)
        screen.blit(text, (screen_width/3, screen_height_notes - 24))

    # print numpy array image on screen
    arr = pygame.surfarray.array3d(screen)  # get the 2d array of RGB values
    for y in range(50):
        for x in range(255):
            for c in range (3):
                arr [y+100,x+820,c] = img [x,y,c]
    pygame.surfarray.blit_array(screen, arr)  # write the array to screen

    # print predicted note on screen
    is_note = False
    current_sound = classes[top_3[0]][9:]
    if current_sound != "other" and str(current_sound).isdigit():
        is_note = True

    if is_note:
        font = pygame.font.SysFont("comicsansms", 72)
        font_color = (0, 128, 0)
        text = font.render(note_names[int(current_sound)], True, font_color)
        screen.blit(text, (screen_width/3 + 90, screen_height_notes + 70))

        font = pygame.font.SysFont("comicsansms", 12)
        font_color = (0, 0, 0)
        for i in range(3):
            text = font.render("{}".format(classes[top_3[i]]) + ": {:4.1f} %".format(proba[0][top_3[i]]*100), True, font_color)
            screen.blit(text, (screen_width / 3*2 + 30, screen_height_notes + 20 + 20*i))

    pygame.display.flip()

    #######################################################################
    # LISTEN TO EVENTS
    #######################################################################

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            key = pygame.key.name(event.key)

            if event.key == pygame.K_SPACE:
                is_playing = False if is_playing == True else True

    #######################################################################
    # DO ACTIONS
    #######################################################################

    if is_playing:
        # play sound
        if is_note:
            sounds[int(current_sound)].play(fade_ms=50)
            pygame.time.delay(650)

        # move on
        current_image = current_image + 1
        if current_image >= num_images:
            current_image = 0
            is_playing = False
