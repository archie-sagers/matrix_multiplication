import numpy as np
import tracemalloc
import time
from scipy.stats import linregress
import matplotlib.pyplot as plt


def strassen(A, B):
    """Multiplys matricies using Stassen Algorithm"""
    n = len(A)
    
    if n <= 2:  # Base case
        return naivemult(A, B)
    
    # Partition matrices into submatrices
    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]
    
    # Recursive multiplication
    P1 = naivemult(A11, B12 - B22)
    P2 = naivemult(A11 + A12, B22)
    P3 = naivemult(A21 + A22, B11)
    P4 = naivemult(A22, B21 - B11)
    P5 = naivemult(A11 + A22, B11 + B22)
    P6 = naivemult(A12 - A22, B21 + B22)
    P7 = naivemult(A11 - A21, B11 + B12)
    
    # Combine results to form C
    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7
    
    # Combine quadrants to form C
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C

def naivemult(A,B):
    """Multiplys matricies using a naive method"""
    (m,n) = np.shape(A)
    C = np.zeros([m,m])
    
    for i in range(m):
        for j in range(m):
            for k in range(m):
                C[i,j] += A[i,k] * B[k,j]
    return C

def memory_usage(somefunc,*args,repeats=2,**kwargs):
    """Finds the space complexity of function"""
    memory = []
    for i in range(repeats):
        tracemalloc.start()
        ans=somefunc(*args,**kwargs)
        x,y = tracemalloc.get_traced_memory()
        memory.append(y)
        tracemalloc.stop()
    mean=np.mean(memory)
    return (mean)
    
def timeit(somefunc,*args,repeats=1,**kwargs):
    """Times the length of a function, repeats this and produces a mean average"""
    times=[]
    for i in range(repeats):
        starttime=time.perf_counter()
        ans=somefunc(*args,**kwargs)
        endtime=time.perf_counter()
        timetaken=endtime-starttime
        times.append(timetaken)
    
    mean=np.mean(times) # compute the mean time taken
    stdev=np.std(times) # compute the standard deviation of times taken (how variable is the time taken)
 
    return (mean)   #The first output will be the mean time taken and 

                 
xs2 = []
ys_naive = []
ys_strassen = []
ys_matmul = []
unsorted = []

for n in range(2, 8):
    xs2.append(n)
    a = np.random.randint(10, size=(2**n, 2**n))
    b = np.random.randint(10, size=(2**n, 2**n))
    mean_mem = memory_usage(naivemult, a, b)
    mean_mem2 = memory_usage(np.matmul, a, b) 
    mean_mem3 = memory_usage(strassen, a, b) 
    ys_naive.append(mean_mem)
    ys_matmul.append(mean_mem2)
    ys_strassen.append(mean_mem3)
    
                        
# Plot graph of results
plt.scatter(np.array(xs2),np.array(ys_strassen),label='Strassen Multiplication') 
plt.scatter(np.array(xs2),np.array(ys_naive),label='Naive Multiplication') 
plt.scatter(np.array(xs2),np.array(ys_matmul),label='Matmul Multiplication') 
plt.legend(loc='upper left')
plt.xlabel('n')
plt.ylabel('Memory')
plt.title('Max memory used to multiply two random matricies')
