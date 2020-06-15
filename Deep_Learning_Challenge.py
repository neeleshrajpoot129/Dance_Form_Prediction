#!/usr/bin/env python
# coding: utf-8

# # HACKEREARTH DEEP LEARNING CHALLENGE

# # Importing The Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Importing Training and Test Data

# In[3]:


df_train=pd.read_csv("train.csv")


# In[4]:


df_train.head()


# In[5]:


df_test=pd.read_csv("test.csv")


# In[6]:


#Check How many Unique Values Are Present
df_train["target"].nunique()
#We Can See That 8 Categories Are Present


# In[7]:


df_test.head()


# # Importing The Deep Learning Libraries

# In[5]:


from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dropout


# # NOW LETS CHECK THE SHAPE AND SIZE OF THE INPUT IMAGE

# In[7]:


from PIL import Image


# In[9]:


im1=Image.open('train/419.jpg')


# In[10]:


im1


# In[11]:


im_array=np.array(im1)


# In[12]:


im_array.shape


# # LOAD ANOTHER RANDOM IMAGE
# 

# In[13]:


im2=Image.open('train/4.jpg')


# In[14]:


im2_array=np.array(im2)


# In[15]:


# WE CAN SEE THAT ALL IMAGES ARE NOT OF EQUAL DIMGENSION
im2_array.shape


# # This is How We Will resize Our Image
# # I am Using I3 Processsor So I am Resizing The Image To (64,64)
# # You Can Resize It higher Dimension Such As (128,128)

# In[16]:


im1=im1.resize(size=(64,64))


# In[17]:


im1


# # LETS PREPROCESS OUR DATASET NOW

# In[18]:


#Checking The Shape Of Dataset
df_train.shape[0]


# In[19]:


for i in range(df_train.shape[0]):
    k=df_train["Image"][i]
    df_train["Image"][i]=Image.open('train/{}'.format(k))
    df_train["Image"][i]=df_train["Image"][i].resize((64,64))
    


# # We Have Resized Our Training Images

# In[20]:


df_train["Image"][0]


# # CONVERTING OUR INPUT SHAPE TO PIXEL FORMAT

# In[22]:


for i in range(df_train.shape[0]):
    k=df_train["Image"][i]
    df_train["Image"][i]=np.array(k)
    


# In[23]:


df_train.head()


# In[ ]:


#CONVERTING OUR INPUT SHAPE TO PIXEL FORMAT


# In[24]:


df_train["Image"][65].shape


# # Here We Can See The Dimension Of Our Input Image 
# # Here (64,64) Are The Height And Width Of Image And 3 Stands For Coloured Picture(RGB).

# In[ ]:


#SAVE THIS PREPROCESSED DATASET INTO ANOTHER CSV FOR FUTURE PURPOSE


# In[25]:


df_train.to_csv("Preprocessed.csv",index=False)


# # WE HAVE PREPROCESSED OUR DATA

# # LETS RESHAPE THEM INTO 4D TENSORS AS THEY REQUIRED FOR INPUT IN KERAS

# In[26]:


y_train=df_train["target"]


# In[27]:


y_train


# In[46]:


X_train=df_train["Image"]


# In[47]:


print(type(X_train))


# In[48]:


for i in range(df_train["Image"].shape[0]):
    X_train[i]=X_train[i].reshape(1,64,64,3)


# # DEFINING THE MODEL

# In[50]:


clf=Sequential()


# In[51]:


clf.add(Conv2D(64,kernel_size=(3,3),input_shape=(64,64,3),activation='relu'))
clf.add(MaxPool2D(pool_size=(2,2)))
clf.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
clf.add(MaxPool2D(2,2))
clf.add(Flatten())
clf.add(Dense(units=128,activation='relu'))
clf.add(Dropout(rate=0.2))
clf.add(Dense(8,activation='softmax'))


# In[52]:


clf.summary()


# In[53]:


clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# # Making X_train and y_train as reauired for our Classifier

# In[38]:


from sklearn.preprocessing  import LabelEncoder
lb=LabelEncoder()


# In[41]:


y_train_scaled=lb.fit_transform(y_train_scaled)


# In[36]:


y_train


# In[42]:


from keras.utils import to_categorical
y_train_scaled = to_categorical(y_train_scaled)


# In[43]:


y_train_scaled.shape


# # NOW VERTCICALLY STACKING UP OUR INPUT ARRAY

# In[49]:


X_train[0].shape


# In[50]:


stacked_array=np.vstack((X_train[0],X_train[1]))


# In[51]:


stacked_array.shape


# In[52]:


#CONTINUING THIS WORK IN LOOP


# In[53]:


for i in range(2,df_train.shape[0]):
    stacked_array=np.vstack((stacked_array,X_train[i]))


# In[54]:


stacked_array.shape


# #  We have Processed Out X_train and y_train to Feed them into the network

# In[55]:


clf.fit(stacked_array,y_train,epochs=12)


# # Checking The Classes

# In[88]:


classes=lb.classes_


# In[57]:


classes


# # NOW MAKING PREDICTIONS

# In[58]:


from keras.preprocessing import image
test_image=image.load_img('test/198.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)


# In[59]:


clf.predict_classes(test_image)


# In[60]:


print("The Prediceted Dance Form is {}".format(classes[3]))


# # MAKING SUBMISSION FILE

# In[61]:


df_test.head()


# # Preprocessing The Test File For Making Predictions

# In[62]:


for i in range(df_test.shape[0]):
    k=df_test["Image"][i]
    df_test["Image"][i]=Image.open('test/{}'.format(k))
    


# In[80]:


df_test["Image"][1]


# In[68]:


for i in range(df_test.shape[0]):
    k=df_test["Image"][i]
    df_test["Image"][i]=k.resize((64,64))
    


# In[81]:


im1=Image.open('test/246.jpg')


# In[82]:


df_test["Image"][1]=im1.resize((64,64))


# In[83]:


df_test.head()


# In[84]:


for i in range(df_test.shape[0]):
    k=df_test["Image"][i]
    df_test["Image"][i]=np.array(k)
    


# In[85]:


df_test.head()


# # Let's Try predict a random image for given dimension
# 

# In[86]:


pre=clf.predict_classes(df_test["Image"][23])


# # WE can see that we need 4 dimenions as we have trained our classifier with 4d

# In[87]:


#Changing the dimensions
X_test=df_test["Image"]


# In[88]:


for i in range(df_test["Image"].shape[0]):
    X_test[i]=X_test[i].reshape(1,64,64,3)


# In[89]:


X_test.to_csv("Preprocessd_test.csv",index=False)


# # Let's Make Predictions
# 

# In[ ]:


l=classes[clf.predict_classes(X_test[4]).shape[0]]
# copying the same code in a loop


# In[ ]:





# In[90]:


pred=[]
for i in range(X_test.shape[0]):
    pred.append(classes[clf.predict_classes(X_test[i]).shape[0]])
    


# In[91]:


pred


# # AS YOU CAN SEE THAT THERE IS SOME SORT OF OVERFIING IN OUR MODEL

# # Let's Make Another Classfier

# In[46]:


clf2=Sequential()


# In[47]:


clf2.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),input_shape=(64,64,3),activation='relu'))


# In[48]:



clf2.add(MaxPool2D(pool_size=(2,2)))
clf2.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
clf2.add(MaxPool2D(2,2))
clf2.add(Flatten())
clf2.add(Dense(units=128,activation='relu'))
clf2.add(Dropout(rate=0.2))
clf2.add(Dense(units=64,activation='relu'))
clf2.add(Dropout(0.2))
clf2.add(Dense(8,activation='softmax'))


# In[49]:


clf2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[166]:


clf2.fit(stacked_array,y_train,epochs=20)


# In[85]:


#created a dummy file for other purposes
name=pd.read_csv("test.csv")


# In[167]:


pred2=[]
for i in range(X_test.shape[0]):
    test_image=image.load_img('test/{}'.format(name["Image"][i]),target_size=(64,64))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    k=clf2.predict_classes(test_image)
    pred2.append(classes[k[0]])


    
    


# In[168]:


pred2


# In[151]:


pred=[]
for i in range(X_test.shape[0]):
    test_image=image.load_img('test/{}'.format(name["Image"][i]),target_size=(64,64))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    k=clf.predict_classes(test_image)
    pred.append(classes[k[0]])


    
    


# In[152]:


pred


# # Making And Saving Submission File

# In[158]:


sample_submission1=pd.DataFrame({"Image":name["Image"],"target":pred})


# In[159]:


sample_submission1.to_csv("sample_submission1.csv",index=False)


# In[169]:


sample_submission2=pd.DataFrame({"Image":name["Image"],"target":pred2})


# In[170]:


sample_submission2.to_csv("sample_submission2.csv",index=False)


# In[171]:


from keras.applications.resnet50 import ResNet50, preprocess_input


# # Reshaping The Size Of Image For ResNet As Minimum Size Requirement is (197,197,3)

# In[5]:


df_train_new=pd.read_csv("train.csv")


# In[6]:


df_test_new=pd.read_csv("test.csv")


# In[8]:


from PIL import Image


# In[9]:


for i in range(df_train_new.shape[0]):
    k=df_train_new["Image"][i]
    df_train_new["Image"][i]=Image.open('train/{}'.format(k))
    df_train_new["Image"][i]=df_train_new["Image"][i].resize((200,200))
    


# In[10]:


for i in range(df_test_new.shape[0]):
    k=df_test_new["Image"][i]
    df_test_new["Image"][i]=Image.open('test/{}'.format(k))
    df_test_new["Image"][i]=df_test_new["Image"][i].resize((200,200))

    


# In[ ]:


from keras.applications.resnet50 import ResNet50, preprocess_input
model = ResQNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))


# In[11]:


for i in range(df_train_new.shape[0]):
    k=df_train_new["Image"][i]
    df_train_new["Image"][i]=np.array(k).astype('float32')
    


# In[12]:


for i in range(df_test_new.shape[0]):
    k=df_test_new["Image"][i]
    df_test_new["Image"][i]=np.array(k).astype('float32')
    


# In[13]:


df_train_new.head()


# In[22]:


X_train_new=df_train_new["Image"]


# In[29]:


X_test_new=df_test_new["Image"]


# In[18]:


X_train_new.shape


# In[23]:


for i in range(df_train_new["Image"].shape[0]):
    X_train_new[i]=X_train_new[i].reshape(1,200,200,3)


# In[225]:


for i in range(df_test_new["Image"].shape[0]):
    X_test_new[i]=X_test_new[i].reshape(1,200,200,3)


# In[ ]:


stacked_array_new=np.vstack((X_train_new[0],X_train_new[1]))


# In[ ]:


stacked_array_test=np.vstack((X_test_new[0],X_test_new[1]))


# In[31]:


for i in range(2,df_train_new.shape[0]):
    stacked_array_new=np.vstack((stacked_array_new,X_train_new[i]))
    


# In[32]:


for i in range(2,df_test_new.shape[0]):
    stacked_array_test=np.vstack((stacked_array_test,X_test_new[i]))
    


# #  My Laptop Is Not Capable Of Training It So Saving These Preprocessed Data into .csv File To Train Them On Google Colab

# In[229]:


colab_train=pd.DataFrame({"Image":X_train_new,"target":df_train_new["target"]})


# In[230]:


colab_train.to_csv("colab_train.csv",index=False)


# In[231]:


colab_test=pd.DataFrame({"Image":X_test_new})
colab_test.to_csv("colab_test.csv",index=False)


# In[2]:


X_train=pd.read_csv("colab_train.csv")


# In[14]:


#checking the dtype 
type(df_train_new["Image"][0])


# In[40]:


#Preprocessing the data, so that it can be fed to the pre-trained ResNet50 model. 
resnet_train_input = preprocess_input(stacked_array_new)


# In[ ]:


#Creating bottleneck features for the training data
train_features = model.predict(resnet_train_input)


# In[ ]:


np.savez('resnet_features_train', features=train_features)


# In[ ]:


model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=(200,200,3)))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])


# In[ ]:


checkpointer = ModelCheckpoint(filepath='scratchmodel.best.hdf5', 
                               verbose=1,save_best_only=True)


# In[ ]:


model.fit(train_features, y_train, batch_size=32, epochs=10,
          validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)


# # Now You Can Make Prediction on Test set

# # Scaling  The Data Taking Higher Dimension
# 

# In[10]:


for i in range(df_train.shape[0]):
    k=df_train["Image"][i]
    df_train["Image"][i]=Image.open('train/{}'.format(k))
    df_train["Image"][i]=df_train["Image"][i].resize((128,128))

    


# In[15]:





# In[18]:


for i in range(df_train.shape[0]):
    df_train["Image"][i]=np.array(df_train["Image"][i])
   


# In[20]:


X_train_scaled=df_train.Image


# In[21]:


for i in range(df_train.shape[0]):
    X_train_scaled[i]=X_train_scaled[i].reshape(1,128,128,3)


# # Scaling Our Inputs

# In[23]:


X_train_scaled=X_train_scaled/255


# In[24]:


y_train_scaled=df_train["target"]


# In[26]:


stacked_scaled=np.vstack((X_train_scaled[0],X_train_scaled[1]))


# In[27]:


for i in range(2,df_train.shape[0]):
    stacked_scaled=np.vstack((stacked_scaled,X_train_scaled[i]))


# In[55]:


clf2.fit(stacked_scaled,y_train_scaled,batch_size=64,epochs=4)


# In[56]:


clf.fit(stacked_scaled,y_train_scaled,batch_size=32,epochs=10)


# In[77]:


for i in range(df_test.shape[0]):
    k=df_test["Image"][i]
    df_test["Image"][i]=Image.open('test/{}'.format(k))
    df_test["Image"][i]=df_test["Image"][i].resize((128,128))

    


# In[78]:


for i in range(df_test.shape[0]):
    df_test["Image"][i]=np.array(df_test["Image"][i])


# In[79]:


X_test_scaled=df_test.Image


# In[80]:


X_test_scaled=X_test_scaled/255


# In[69]:


for i in range(df_test.shape[0]):
    #df_train["Image"][i]=np.array(df_train["Image"][i])
    #df_train["Image"][i]=df_train.reshape(1,128,128,3)
    X_test_scaled[i]=X_train_scaled[i].reshape(1,128,128,3)


# In[70]:


X_test_scaled.shape


# In[72]:


stacked_scaled_test=np.vstack((X_test_scaled[0],X_test_scaled[1]))


# In[73]:


for i in range(2,df_test.shape[0]):
    stacked_scaled_test=np.vstack((stacked_scaled_test,X_test_scaled[i]))


# In[ ]:





# In[83]:


from keras.preprocessing import image


# In[89]:


pred2=[]
for i in range(df_test.shape[0]):
    test_image=image.load_img('test/{}'.format(name["Image"][i]),target_size=(128,128))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    k=clf2.predict_classes(test_image)
    pred2.append(classes[k[0]])


    
    


# In[90]:


pred2


# In[91]:


pred1=[]
for i in range(df_test.shape[0]):
    test_image=image.load_img('test/{}'.format(name["Image"][i]),target_size=(128,128))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    k=clf.predict_classes(test_image)
    pred1.append(classes[k[0]])


    
    


# In[92]:


pred1


# In[93]:


sample_submission3=pd.DataFrame({"Image":name["Image"],"target":pred1})


# In[94]:


sample_submission3.to_csv("sample_submission3.csv",index=False)


# In[95]:


sample_submission4=pd.DataFrame({"Image":name["Image"],"target":pred2})


# In[96]:


sample_submission4.to_csv("sample_submission4.csv",index=False)

