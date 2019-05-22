import numpy as np
import cv2
import h5py
import os
# import matplotlib.pyplot as plt
import tqdm  # Optional

# matplotlib inline

# Prepare reading datasets, metadata texts
prefix = "landmark_aligned_face."
metadata = ['fold_0_data.txt', 'fold_1_data.txt', 'fold_2_data.txt', 'fold_3_data.txt', 'fold_4_data.txt']
classes = ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(25, 32)", "(38, 43)", "(48, 53)", "(60, 100)"]

# Since there are labels that do not match the classes stated, need to fix them
classes_to_fix = {'35': classes[5], '3': classes[0], '55': classes[7], '58': classes[7],
                  '22': classes[3], '13': classes[2], '45': classes[5], '36': classes[5],
                  '23': classes[4], '57': classes[7], '56': classes[6], '2': classes[0],
                  '29': classes[4], '34': classes[4], '42': classes[5], '46': classes[6],
                  '32': classes[4], '(38, 48)': classes[5], '(38, 42)': classes[5], '(8, 23)': classes[2],
                  '(27, 32)': classes[4]}

none_count = 0  # Still there are unlabeled images in the datasets, we count them here


def return_folder_info(textfile):
    global none_count
    # one big folder list
    folder = []
    # start processing metadata txt
    with open(textfile) as text:
        lines = text.readlines()
        for line in lines[1:]:
            line = line.strip().split("\t")
            # real image path
            img_path = 'aligned/' + line[0] + "/" + prefix + line[2] + "." + line[1]
            if line[3] == "None":
                none_count += 1
                continue
            else:
                # We store useful metadata infos
                folder.append([img_path] + line[3:5])
                if folder[-1][1] in classes_to_fix:
                    folder[-1][1] = classes_to_fix[folder[-1][1]]
    return folder


# Now we get infos for all 5 cross validation folders
all_folders = []
for textfile in metadata:
    folder = return_folder_info(textfile)
    all_folders.append(folder)
print("A sample:", all_folders[0][0])
print("No. of Pics without Age Group Label:", none_count)


# Methods for processing img arrays and one-hot generation
def imread(path, width, height):
    img = cv2.imread(path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    return img


def build_one_hot(age):
    label = np.zeros(len(classes), dtype=int)
    label[classes.index(age)] = 1
    return label


# image size for VGGFace
width, height = 336, 336

# loop for reading imgs from five folders
all_data = []
all_labels = []
print("Start reading images data...")
for folder in all_folders:
    data = []
    labels = []
    for i in tqdm.tqdm(range(len(folder))):  # here using tqdm to monitor progress
        img = imread(folder[i][0], width, height)
        one_hot = build_one_hot(folder[i][1])
        data.append(img)
        labels.append(one_hot)
    all_data.append(data)
    all_labels.append(labels)
    print("One folder done...")
print("All done!")
# plt.subplot(151),plt.imshow(all_data[0][0]),plt.title(all_labels[0][0])
# plt.subplot(152),plt.imshow(all_data[1][0]),plt.title(all_labels[1][0])
# plt.subplot(153),plt.imshow(all_data[2][0]),plt.title(all_labels[2][0])
# plt.subplot(154),plt.imshow(all_data[3][0]),plt.title(all_labels[3][0])
# plt.subplot(155),plt.imshow(all_data[4][0]),plt.title(all_labels[4][0])


# calculation of channel-wise BGR means for five folders
b_folders = []
g_folders = []
r_folders = []
n_images_folders = []

# First we summarize rgb values
for i in tqdm.tqdm(range(0, 5)):
    b = np.zeros((height, width))
    g = np.zeros((height, width))
    r = np.zeros((height, width))
    for img in all_data[i]:
        b += img[:, :, 0]
        g += img[:, :, 1]
        r += img[:, :, 2]
    b_folders.append(b)
    g_folders.append(g)
    r_folders.append(r)
    n_images_folders.append(len(all_data[i]))

# Then we generate BGR mean for each cross validation situation
# eg. When we validate folder 1, RGB mean will be generated from folder 2~5
bgr_means = []
for i in range(0, 5):
    folders = [0, 1, 2, 3, 4]
    folders.remove(i)
    b = np.zeros((height, width))
    g = np.zeros((height, width))
    r = np.zeros((height, width))
    n_image = 0
    for folder_index in folders:
        b += b_folders[folder_index]
        g += g_folders[folder_index]
        r += r_folders[folder_index]
        n_image += n_images_folders[folder_index]
    bgr_means.append(np.array([np.mean(b / n_image), np.mean(g / n_image), np.mean(r / n_image)]))

print("BGR Means Array:", bgr_means)

# Generate mean image for each cross validation situation
# eg. When we validate folder 1, RGB mean will be generated from folder 2~5
mean_imgs = []
for i in tqdm.tqdm(range(0, 5)):
    folders = [0, 1, 2, 3, 4]
    folders.remove(i)
    mean_image = np.zeros(all_data[0][0].shape)
    n_image = 0
    for folder_index in folders:
        for img in all_data[folder_index]:
            mean_image += img
        n_image += n_images_folders[folder_index]
    mean_imgs.append(np.array(mean_image / n_image, dtype=np.uint8))

print(mean_imgs[0].shape, mean_imgs[1].dtype)
# plt.subplot(121),plt.imshow(mean_imgs[0])
# plt.subplot(122),plt.imshow(mean_imgs[1])


# Generate h5py dataset
with h5py.File('faces_dataset.h5', 'w') as f:
    for i in range(0, 5):
        dset_face = f.create_dataset("data_" + str(i + 1), data=np.array(all_data[i]))
        dset_headers = f.create_dataset('labels_' + str(i + 1), data=np.array(all_labels[i]))
    dst_bgr_means = f.create_dataset('bgr_means', data=np.array(bgr_means))
    # dst_mean_imgs = f.create_dataset('mean_imgs', data = np.array(mean_imgs))
print("Generation Success!")

