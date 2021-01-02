#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::


#         [5, 6, 7, 8, 9]])

# In[1]:


import numpy as np
arr = np.arange(10)
arr


# In[3]:


arr.reshape(2,5)


# In[ ]:


OR


# In[4]:


arr = np.arange(10)
arr


# In[5]:


arr.reshape(2,-1)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[6]:


a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)


# In[7]:


# Method 1:
np.concatenate([a, b], axis=0)


# In[8]:


# Method 2:
np.vstack([a, b])


# In[9]:


# Method 3:
np.r_[a, b]


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[18]:


a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)


# In[19]:


a


# In[20]:


b


# In[15]:


# Method 1:
np.concatenate([a, b], axis=1)


# In[16]:


# Method 2:
np.hstack([a, b])


# In[17]:


# Method 3:
np.c_[a, b]


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[22]:


import numpy as np 
arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)


# In[23]:


arr1


# In[24]:


arr2


# In[25]:


arr3


# In[26]:


array_of_arrays = np.array([arr1, arr2, arr3])
print('array_of_arrays: ', array_of_arrays)


# In[27]:


arr_2d = np.concatenate(array_of_arrays)
print(arr_2d)


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[2]:


import numpy as np
arr1 = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]])
arr1


# In[8]:


arr1.ravel()


# In[9]:


# or 
arr1.flatten()


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[85]:


arr = np.arange(15)
arr


# In[86]:


arr.reshape(5,3)


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[28]:


x = np.ones((5,5))
x


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[38]:


arr = np.random.randn(5, 6)

arr


# In[41]:


arr.mean()


# In[43]:


# or
np.mean(arr)


# In[13]:


# or
arr1 = np.arange(1,31).reshape(5,6)
arr1


# In[15]:


arr1.mean()


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[100]:


arr = np.random.randn(5, 6)
arr


# In[101]:


np.std(arr)


# In[16]:


# or
arr1 = np.arange(1,31).reshape(5,6)
arr1


# In[17]:


np.std(arr1)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[96]:


arr = np.random.randn(5, 6)
arr


# In[97]:


np.median(arr)


# In[22]:


# or
arr1 = np.arange(1,31).reshape(5,6)
arr1


# In[23]:


np.median(arr1)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[92]:


arr = np.random.randn(5, 6)

arr


# In[95]:


np.transpose(arr)


# In[24]:


# or
arr1 = np.arange(1,31).reshape(5,6)
arr1


# In[25]:


np.transpose(arr1)


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[49]:


arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
arr


# In[50]:


arr1 = np.diag(arr)
arr1


# In[51]:


arr1.sum()


# In[52]:


np.sum(arr1)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[24]:


arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
arr


# In[25]:


arr1 = np.linalg.det(arr)
arr1


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[64]:


arr = np.arange(10)
arr


# In[66]:


np.percentile(arr,5)


# In[67]:


np.percentile(arr,95)


# ## Question:15

# ### How to find if a given array has any null values?

# In[122]:


import numpy as np
Z = np.random.randint(0,3,(3,10))
Z


# In[123]:


Z.any()

