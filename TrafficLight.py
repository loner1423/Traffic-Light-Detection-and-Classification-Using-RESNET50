from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os

import cv2
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint
import keras

import tensorflow as tf
from PIL import Image
from os import path
from utils import label_map_util
from sklearn.metrics import confusion_matrix
import seaborn as sns


main = tkinter.Tk()
main.title("Traffic Lights Detection and Classification using Resnet50")
main.geometry("1300x1200")

class_labels = ['Go', 'Go Forward', 'Go Left', 'Stop', 'Stop Left', 'Warning', 'Warning Left']

global filename
global X, Y
global X_train, X_test, y_train, y_test
global resnet_model, filename
labels = []

for root, dirs, directory in os.walk("Dataset"):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    global labels
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    global filename, X, Y
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Dataset Path Loaded\n\n')
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))
                    X.append(img)
                    label = getLabel(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)                    
    text.insert(END,'Total images found in Dataset : '+str(X.shape[0])+"\n")        
    text.insert(END,"Class labels found in dataset : "+str(class_labels))
    #visualizing class labels count found in dataset
    names, count = np.unique(Y, return_counts = True)
    height = count
    bars = class_labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (7, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def processDataset():
    global X, Y
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Dataset images Normalization & Processing Completed\n")

def trainTest():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Image Processing & Normalization Completed\n\n")
    text.insert(END,"80% images used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% image used to train algorithms : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, y_test):
    global class_labels
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")    
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = class_labels, yticklabels = class_labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(class_labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()        

def trainResnet():
    global X_train, X_test, y_train, y_test, resnet_model
    text.delete('1.0', END)
    resnet_model = ResNet50(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in resnet_model.layers:
        layer.trainable = False
    resnet_model = Sequential()
    resnet_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (2, 2)))
    resnet_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (2, 2)))
    resnet_model.add(Flatten())
    resnet_model.add(Dense(units = 256, activation = 'relu'))
    resnet_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
    if os.path.exists("model/resnet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = resnet_model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/resnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        resnet_model.load_weights("model/resnet_weights.hdf5")
    #perform prediction test data    
    predict = resnet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("Resnet50", predict, y_test1)#calculate accuracy and other metrics

#use this function to predict fish species uisng extension model
def predictLight(image):
    global resnet_model
    resnet_model = load_model("model/resnet_weights.hdf5")
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32,32))#resize image
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)#convert image as 4 dimension
    img = np.asarray(im2arr)
    img = img.astype('float32')#convert image features as float
    img = img/255 #normalized image
    predict = resnet_model.predict(img)#now predict dog breed
    predict = np.argmax(predict)
    print(predict)
    return class_labels[predict]

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def read_traffic_lights(image, boxes, scores, classes, max_boxes_to_draw=20, min_score_thresh=0.2, traffic_ligth_label=10):
    crop_img = None
    im_width, im_height = image.size
    red_flag = False
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        print(str(scores[i])+" "+str(classes[i]))
        if scores[i] > min_score_thresh and classes[i] == traffic_ligth_label:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            crop_img = np.asarray(image.crop((left, top, right, bottom)))
            print("==="+str(crop_img.shape))
    return crop_img 

def imageClassification():
    global resnet_model
    filename = filedialog.askopenfilename(initialdir = "testImages")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('model/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap('model/mscoco_label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = Image.open(filename)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            img = read_traffic_lights(image, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))
            if img is not None:
                predictedLabel = predictLight(img)
                image = cv2.imread(filename)
                image = cv2.resize(image, (600,400))#display image with predicted output
                cv2.putText(image, 'Predicted As : '+predictedLabel, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
                cv2.imshow("Predicted Light", image)
                cv2.imshow("Detected Light", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                cv2.waitKey(0)
                #break
            else:
                image = cv2.imread(filename)
                image = cv2.resize(image, (600,400))#display image with predicted output
                cv2.putText(image, 'Unable to detect & Classify', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
                cv2.imshow("Predicted Light", image)
                cv2.waitKey(0)
                #break
    print("done")
    
def videoClassification():
    global resnet_model
    filename = filedialog.askopenfilename(initialdir = "Videos")
    cap = cv2.VideoCapture(filename)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('model/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap('model/mscoco_label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            while True:
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.resize(frame, (600, 600))
                    h, w, c = frame.shape
                    cv2.imwrite("test.jpg", frame)
                    image = Image.open("test.jpg")
                    image_np = load_image_into_numpy_array(image)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
                    img = read_traffic_lights(image, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))
                    if img is not None:
                        predictedLabel = predictLight(img)
                        cv2.putText(frame, 'Predicted As : '+predictedLabel, (100, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
                        cv2.imshow("Detected Light", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    cv2.imshow('Video Output', frame)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break
                else:
                    break
    cap.release()
    cv2.destroyAllWindows()
            
def values(filename, acc, loss):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values[acc]
    loss_value = train_values[loss]
    return accuracy_value, loss_value    
    
def graph():
    val_acc, val_loss = values("model/resnet_history.pckl", "val_accuracy", "val_loss")
    acc, loss = values("model/resnet_history.pckl", "accuracy", "loss")    
    plt.figure(figsize=(6,4))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(loss)
    plt.plot(val_acc)
    plt.plot(val_loss)
    plt.legend(['Train Accuracy', 'Train Loss','Validation Accuracy','Validation Loss'], loc='upper left')
    plt.title('Resnet50 Algorithm Training & Validation Accuracy, Loss Graph')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Traffic Lights Detection and Classification using Resnet50')
title.config(bg='chocolate', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Traffic Light Dataset", command=uploadDataset)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='dodger blue')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=700,y=200)
processButton.config(font=font1)

splitButton = Button(main, text="Train & Test Split", command=trainTest)
splitButton.place(x=700,y=250)
splitButton.config(font=font1) 

resnetButton = Button(main, text="Train Resnet50 Algorithm", command=trainResnet)
resnetButton.place(x=700,y=300)
resnetButton.config(font=font1)

imageButton = Button(main, text="Image Classification", command=imageClassification)
imageButton.place(x=700,y=350)
imageButton.config(font=font1)

videoButton = Button(main, text="Video Classification", command=videoClassification)
videoButton.place(x=700,y=400)
videoButton.config(font=font1)

graphButton = Button(main, text="Resnet50 Training Graph", command=graph)
graphButton.place(x=700,y=450)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='light salmon')
main.mainloop()
