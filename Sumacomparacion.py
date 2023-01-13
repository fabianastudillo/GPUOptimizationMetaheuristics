from theano import *
import cupy as cp
import time 
import numpy as np
from cupy import dot
from numba import guvectorize, int64, void, float64
from numba import cuda
import pkg_resources, os

import os
os.environ['NUMBAPRO_NVVM']      = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\nvvm\bin\nvvm64_31_0.dll'
os.environ['NUMBAPRO_LIBDEVICE'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\nvvm\libdevice'

stream = cuda.stream()
# MATRIZ CUADRADA 


dim = 1000
x = np.random.randint(dim,size=(dim,dim))
y = np.random.randint(dim,size=(dim,dim))
A = np.random.rand(dim,dim)*1000
A = np.round(A)
w = cp.asarray(x)
z = cp.asarray(y)
cp.cuda.Stream.null.synchronize()

print("Comparacion suma elemento a elemento")

#SUMA EN THEANO 
start = time.time()
#a = theano.tensor.vector() # declare variable
#
a = tensor.dmatrix()
out = a+a # build symbolic expression
f = theano.function([a], out) # compile function
out1 = f(x)
print(out1)
end = time.time()
print("Tiempo con theano suma = %s" % (end - start))

#SUMA EN CUPY 
start = time.time()
#out2= cp.multiply(w,w)
out2= w + w
cp.cuda.Stream.null.synchronize()
print(out2)
end = time.time()
print("Tiempo con cupy suma = %s" % (end - start))

#SUMA EN NUMBA
@guvectorize([void(float64[:,:], float64[:,:], float64[:,:])], '(m,n),(n,p)->(m,p)', target='cuda')
def matsum(A,B,C):
        m,n = A.shape
        n,p = B.shape 
        for i in range(m):
            for j in range(p): 
                C[i,j] = A[i,j]+B[i,j]
                      
#matsum.max_blocksize = 32    
d_x = cuda.to_device(A, stream=stream)

start = time.time()
C = matsum(A,A)
print(C)
end = time.time()
print("Tiempo con guvectorize (numba) suma = %s" % (end - start))

#SUMA EN NUMPY

start = time.time()
#out3= np.multiply(x,x)
out3= x + x
print(out3)
end = time.time()
print("Tiempo con Numpy suma = %s" % (end - start))


