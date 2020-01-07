from imagecluster import calc, io as icio, postproc
import shutil
import os

#######################################################################
# SETTING
#######################################################################

data_root = "C:\\Data\\Dev-Data\\music\\"
if os.getenv("ROCKETMAN_DATA") is not None:
    data_root = os.getenv("ROCKETMAN_DATA")

fpath = data_root
fpath_notes = fpath + "notes\\"
cpath_notes = fpath + "cluster\\"
similarity = 0.65

##################################################################################
# CLUSTER IMAGES FOR EASY LABELING
print ("CLUSTER IMAGES FOR EASY LABELING")
##################################################################################

# Create image database in memory. This helps to feed images to the NN model
# quickly.
ias = icio.read_images(fpath_notes, size=(50, 250))

# Create Keras NN model.
model = calc.get_model()

# Feed images through the model and extract fingerprints (feature vectors).
fps = calc.fingerprints(ias, model)

# Optionally run a PCA on the fingerprints to compress the dimensions. Use a
# cumulative explained variance ratio of 0.95.
#fps = calc.pca(fps, n_components=0.95)

# Run clustering on the fingerprints.  Select clusters with similarity index
clusters = calc.cluster(fps, sim=similarity)

c = 0
for lists in clusters.values():
    for items in lists:
        print("--- Cluster: ", c)

        for item in items:
            filepath = item
            filepathsplit = item.split("\\")
            filename = filepathsplit[-1]
            filepath_c = cpath_notes + "c" + str(c) + "_" + filename
            print("------ "+filepath_c)
            shutil.copy(filepath, filepath_c)
        c = c + 1

print ("DONE!")

