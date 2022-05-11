#!/usr/bin/env python
# coding: utf-8

# # Hashing

# In[5]:


# Importing the necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[8]:


# Reading multiple images from a folder
mydir = os.getcwd()
data_dir = os.path.join(mydir,'yalefaces')
#print(data_dir)
imgs = os.listdir(data_dir)
images = []
for file in imgs:
    file_path = os.path.join(data_dir,file)
    img = mpimg.imread(file_path)
    if img is not None:
        images.append(img)
print(len(images))


# In[9]:


# Vectorizing the images and storing it in a list
imgs_vec = []
for image in images:
    row,col = image.shape
    img_vec = image.reshape(row*col)
    img_vec_norm = img_vec/(np.linalg.norm(img_vec))
    imgs_vec.append(img_vec_norm)
print(len(imgs_vec))
print(row*col)


# ## Locality Sensitive Hashing – Random Projections

# In[10]:


# Generator Function to generate random unit vectors for Hashing
def genRandomHashVec(m,length):
    hash_vec = []
    for i in range(m):
        v = np.random.uniform(-1,1,length)
        v_ =  v/(np.linalg.norm(v))
        hash_vec.append(v_)
    return hash_vec


# In[11]:


# Function For Local Sensitive Hashing
def LSH(hash_vec,data_pt):
    hash_code = []
    for i in range(len(hash_vec)):
        if np.dot(data_pt,hash_vec[i])>0:
            hash_code.append('1')
        else:
            hash_code.append('0')
    return hash_code 


# In[12]:


# Generate 10 random vectors of the same size of the image vector
hash_vector = genRandomHashVec(10,len(imgs_vec[0]))
print(len(hash_vector))


# In[13]:


# Test the LSHash function 
LSH(hash_vector,imgs_vec[0])


# In[14]:


# Creating a Image Dictionary using the hash as the keys
image_dict = {}
for i in range(len(imgs_vec)):
    hash_code = LSH(hash_vector,imgs_vec[i])
    str_hash_code = ''.join(hash_code)
    if str_hash_code not in image_dict.keys():
        image_dict[str_hash_code] = [i]
    else:
        image_dict[str_hash_code].append(i)


# In[15]:


# Displaying the Hashes
cols_names = ['Hash_codes','Image_Index']
df = pd.DataFrame(image_dict.items(),columns = cols_names)
df.head(30)


# In[16]:


# Getting the keys and values of the Dictionary
keys = list(image_dict.keys())
values = list(image_dict.values())


# In[17]:


# Plotting images with same hash code
igs = [images[i] for i in range(len(images)) if i in values[2]]
fig = plt.figure()
cols = 2
n_images = len(igs)
for n,image in zip(range(n_images),igs):
    ax = fig.add_subplot(cols,np.ceil(n_images/float(cols)),n+1)
    plt.gray()
    plt.imshow(image)
fig.set_size_inches(np.array(fig.get_size_inches())*n_images)
plt.show()


# In[4]:


os.getcwd()


# In[ ]:




