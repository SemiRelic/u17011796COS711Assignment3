# Estienne du Toit      u17011796

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import tensorflow as tf
import numpy as np

import time

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

filename = "Train.csv"

image_dim = 160

trainData = []
for x in range(3906):
	trainData.append([])

try:
    with open(filename, 'r') as f:
        tmp = f.readline()
        # print(tmp)
        for i in range(3906):
            line = f.readline()
            # print(line)
            prev = 0
            nxt = 0
            for j in range(6):
                done = False
                while(done == False):
                	if(nxt == len(line)-1):
                		trainData[i].append(float(line[prev:nxt]))
                		done=True
                	elif(line[nxt]==","):
                		# print(line[prev:nxt])
                		if(j<2):
                			trainData[i].append(line[prev:nxt])
                		else:
                			trainData[i].append(float(line[prev:nxt]))
                		prev = nxt + 1
                		done = True
                	nxt+=1
                # end while
            # end for
            # print(trainData[i])
        # end for
    f.close()

except IOError as ioe:
    print("The file " + filename + " cannot be found. ")


wholeImages = []
wholeTargets = []


for x in range(len(trainData)):
    if(x % 1000 == 0):
        print(x)
    filename = "Train_Images/" + trainData[x][0] + ".jpg"
    img = load_img(filename)

    # print(wholeImages[0])
    # img.show()
    
    
    img1 = img.crop((trainData[x][2],trainData[x][3],trainData[x][4]+trainData[x][2],trainData[x][5]+trainData[x][3]))
    
    
    # img1.show()

    # img2 = img_to_array(img1)

    dim = (image_dim,image_dim)
    img2 = img1.resize(dim)

    # img2.show()

    img_array = img_to_array(img2)
    wholeImages.append(img_array)

    wholeTargets.append([])

    if(trainData[x][1] == 'fruit_healthy'):
        wholeTargets[x].append(1.0)
        wholeTargets[x].append(0.0)
        wholeTargets[x].append(0.0)
    elif(trainData[x][1] == 'fruit_woodiness'):
        wholeTargets[x].append(0.0)
        wholeTargets[x].append(1.0)
        wholeTargets[x].append(0.0)
    else:
        wholeTargets[x].append(0.0)
        wholeTargets[x].append(0.0)
        wholeTargets[x].append(1.0)

# print(wholeTargets[0])
# print(wholeTargets[500])

trainImages = []
# print(len(wholeImages))#3906
for x in range(2906):
    trainImages.append(wholeImages[x])

testImages = []
for x in range(1000):
    testImages.append(wholeImages[x])

trainTargets = []
# print(len(wholeTargets))#3906
for x in range(2906):
    trainTargets.append(wholeTargets[x])

testTargets = []
for x in range(1000):
    testTargets.append(wholeTargets[x])


trainImages = np.array(trainImages)
# print(trainImages[0])
trainTargets = np.array(trainTargets)
# print(trainTargets[0])
testImages = np.array(testImages)
# print(testImages[0])
testTargets = np.array(testTargets)
# print(testTargets[0])


for runs in range(10):
    tf.keras.backend.clear_session()

    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(image_dim,image_dim,3)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    start_time = time.time()

    filename='LogOfRun'+str(runs+1)+'.csv'
    history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

    history = model.fit(trainImages, trainTargets, epochs=50, callbacks=[history_logger])


    pred = model.predict(testImages)

    # print(pred)

    results = open("ResultsOfRun"+str(runs+1)+".txt","a")
    # results.write("Image_ID,class\n")


    # results.write("id"+"class"+"\n")

    correct = 0
    for x in range(len(testTargets)):
        maxm = 0
        classV = 0
        for j in range(3):
            if(pred[x][j] > maxm):
                maxm = pred[x][j]
                classV = j

        classS = ""
        if(classV == 0):
            classS = "fruit_healthy"
            if(testTargets[x][0] == 1.0):
                correct+=1
        elif(classV == 1):
            classS = "fruit_woodiness"
            if(testTargets[x][1] == 1.0):
                correct+=1
        else:
            classS = "fruit_brownspot"
            if(testTargets[x][2] == 1.0):
                correct+=1


        # results.write(compData[x]+","+classS+","+str(0.5)+","+str(0)+","+str(0)+","+str(512)+","+str(512)+"\n")

    # print(correct)
    print("Test Accurracy: " + str((correct/1000) * 100))
    results.write("Test Accurracy: " + str((correct/1000) * 100)+"\n")

    t = "%s" % (time.time() - start_time)
    results.write("Time: "+t + "\n")

    results.close()

#end runs
