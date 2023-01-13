from theano import *
import cupy as cp
import time 
import numpy as np
from cupy import dot
from numba import guvectorize

# MATRIZ CUADRADA 

dim = 100
x = np.random.randint(dim,size=(dim,dim))
y = np.random.randint(dim,size=(dim,dim))
A = np.random.rand(dim,dim)*1000
A = np.round(A)
w = cp.asarray(x)
z = cp.asarray(y)
cp.cuda.Stream.null.synchronize()
#

print("Comparacion multiplicacion elemento a elemento")

#MULTIPLICACION EN NUMBA
start = time.time()
#a = theano.tensor.vector() # declare variable
#
a = tensor.dmatrix()
out = a*a # build symbolic expression
f = theano.function([a], out) # compile function
end = time.time()
out1 = f(x)
print(out1)

print("Tiempo con theano multiplicacion = %s" % (end - start))

#MULTIPLICACION EN CUPY
start = time.time()
out2= cp.multiply(w,w)
cp.cuda.Stream.null.synchronize()
end = time.time()
print(out2)
print("Tiempo con cupy multiplicacion = %s" % ((end - start)))

#MULTIPLICACION EN NUMBA
#Guvectorize

@guvectorize(['void(int64[:,:], int64[:,:], int64[:,:])'], '(m,n),(n,p)->(m,p)',target = 'cuda')
def matmul(A,B,C):
        m,n = A.shape
        n,p = B.shape 
        for i in range(m):
            for j in range(p): 
                C[i,j] = A[i,j] * B[i,j]
                
                
start = time.time()
C = matmul(A,A)
end = time.time()
print(C)
print("Tiempo con guvectorize (numba) multiplicacion= %s" % (end - start))

#MULTIPLICACION EN NUMPY

start = time.time()
out3= np.multiply(x,x)
end = time.time()
print(out3)
print("Tiempo con Numpy multiplicacion = %s" % (end - start))


