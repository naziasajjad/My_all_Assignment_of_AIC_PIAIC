#!/usr/bin/env python
# coding: utf-8

# # **Assignment 1 For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[292]:


import numpy as np


# 2. Create a null vector of size 10 

# In[1]:


import numpy as np
a = np. zeros(10)
a


# 3. Create a vector with values ranging from 10 to 49

# In[2]:


import numpy as np
z = np.arange(10,50)
z


# 4. Find the shape of previous array in question 3

# In[4]:


import numpy as np
z = np.arange(10,50)
z


# In[288]:


z.shape


# 5. Print the type of the previous array in question 3

# In[5]:


import numpy as np
z = np.arange(10,50)
z


# In[286]:


z.dtype


# 6. Print the numpy version and the configuration
# 

# In[5]:


print(np.__version__)
np.show_config()


# 7. Print the dimension of the array in question 3
# 

# In[6]:


import numpy as np
z = np.arange(10,50)
z


# In[7]:


z.ndim


# 8. Create a boolean array with all the True values

# In[7]:


import numpy as np 
a= np.array([1,3,11,5,9,7,])    
a


# In[9]:


print(a[a>5])


# # OR

# In[4]:


arr = np.array([[1, 2, 1], [4, 3, 6]])
arr


# In[5]:


arr2 = np.array([[2, 4, 8], [7, 5, 12]])
arr2


# In[6]:


y= arr2 > arr
y


# In[7]:


y.dtype


# 9. Create a two dimensional array
# 
# 
# 

# In[8]:


import numpy as np
a = np.array([[10,20,30],[40,50,60],[70,80,90]])
a


# In[9]:


a.ndim


# 10. Create a three dimensional array
# 
# 

# In[10]:


import numpy as np
x = np.array([[[10,20,30],[40,50,60],[50,30,85],[70,80,90]]])
x


# In[11]:


x.ndim


# In[8]:


import numpy as np
a = np.array([[[10,20,30,40,50],[60,70,80,90,100],[110,120,130,140,150]]])
a


# In[9]:


a.ndim


# In[ ]:


or


# In[15]:


import numpy as np
b = np.array([[[10,20,30,40,50],[60,70,80,90,100],[110,120,130,140,150],[160,170,180,190,200]]])
b


# In[16]:


b.ndim


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[271]:


import numpy as np
Z = np.arange(40)
Z


# In[272]:


Z = Z[::-1]


# In[273]:


Z


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[17]:


y = np.zeros(10)
y


# In[19]:


y[4] = 1
y


# 13. Create a 3x3 identity matrix

# In[20]:


a = np.eye(3)
a


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[263]:


import numpy as np
arr = np.array([1, 2, 3, 4, 5])


# In[264]:


arr


# In[265]:


arr.dtype


# In[21]:


arr1 = arr.astype('float64')
# arr1 = arr.sttype(np.float64)


# In[22]:


arr1


# In[23]:


arr1.dtype


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[259]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  

arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])


# In[256]:


arr1


# In[257]:


arr2


# In[258]:


arr1*arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[243]:


arr1 = np.array([[1., 2., 3.],
                    [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.], 
                    [7., 2., 12.]])

       


# In[244]:


arr1


# In[245]:


arr2


# In[252]:


arr3 = arr1 == arr2
arr3


# In[ ]:


or


# In[249]:


arr3 = np.intersect1d(arr1,arr2)
arr3


# In[ ]:


or


# In[251]:


arr3 = np.in1d(arr1,arr2)
arr3


# 17. Extract all odd numbers from arr with values(0-9)

# In[89]:


arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr


# In[90]:


arr[arr % 2 == 1]


# 18. Replace all odd numbers to -1 from previous array

# In[91]:


arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr


# In[93]:


arr[arr % 2 == 1] = -1
arr


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[32]:


arr = np.arange(10)
arr


# In[33]:


arr[5:8] = 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[39]:


z = np.ones((5,5))
z


# In[40]:


z[1:-1,1:-1] = 0
z


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# # Replace the value 5 to 12

# In[82]:


arr2d = np.array([[1, 2, 3],

                    [4, 5, 6], 

                    [7, 8, 9]])
arr2d


# In[83]:


arr2d[1, 1] = 12
arr2d


# In[85]:


# or
arr2d[1][1] = 12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[86]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d


# In[88]:


arr3d[0] = 64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[95]:


arr2d = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr2d


# In[24]:


# or
arr2d = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr2d


# In[96]:


arr2d[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[241]:


arr1 = np.arange(9).reshape(3,3)
arr1


# In[27]:


arr1[1:,:-2]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[95]:


arr2d = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr2d


# In[59]:


arr2d[:2, 0:]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[234]:


Z = np.random.random((10,10))
Z


# In[235]:


Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[10]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])


# In[11]:


a


# In[12]:


b


# In[17]:


np.intersect1d(a,b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[13]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])


# In[14]:


a


# In[15]:


b


# In[16]:


np.where(a == b)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[218]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)


# In[219]:


names


# In[220]:


data 


# In[221]:


data[names != 'Will'] = 7
data


# In[226]:


data[names != 'Joe'] = 7
data


# In[ ]:





# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[137]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(names,data)


# In[132]:


names != 'Joe'


# In[227]:


names != 'Will'


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[20]:


import numpy as np
a = np.arange(15).reshape(5,3)
a


# In[ ]:


or


# In[21]:


arr = np.arange(15).reshape(5,3)
arr = np.random.uniform(1,16, size=(5,3))
print(arr)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[34]:


import numpy as np 
x = np.arange(1,17).reshape(2,2,4)
x


# 33. Swap axes of the array you created in Question 32

# In[65]:


import numpy as np 
x = np.arange(1,17).reshape(2,2,4)
x


# In[69]:


x[[0,1]] = x[[1,0]]
x


# In[70]:


#or
x[[0,1], :-2]
x


# In[71]:


#or
x.swapaxes(1, 2)
x


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[199]:


arr = np.arange(10)


# In[200]:


arr


# In[201]:


np.sqrt(arr)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[176]:


x = np.random.randn(12)
y = np.random.randn(12)


# In[177]:


x


# In[178]:


y


# In[179]:


np.maximum(x, y)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[172]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names


# In[173]:


np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[72]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
print(a,b)


# In[171]:


np.setdiff1d(a,b)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[136]:


import numpy as np
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray


# In[152]:


np.delete(sampleArray, 2,1)


# In[144]:


newColumn = np.array([[10,10,10]])
newColumn
                       


# In[167]:


np.insert(sampleArray, 2, 10, axis = 1)


# In[165]:


np.insert(sampleArray,2,10, axis =0)


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[195]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])


# In[196]:


x


# In[197]:


y


# In[198]:


x.dot(y)


# In[ ]:


OR


# In[193]:


z=np.dot(x,y)
z


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[90]:


arr = np.random.randint(3,size=20)
arr


# In[91]:


arr.cumsum()


# In[ ]:


OR


# In[188]:


z = np.random.randn(20)


# In[189]:


z


# In[190]:


z.cumsum()


# In[ ]:




