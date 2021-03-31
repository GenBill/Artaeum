import numpy as np
def prime_gener(np_Length):
    # the Sieve of Eratosthenes
    nums_list = np.zeros([np_Length,2], dtype = int)
    nums_label = np.ones([np_Length], dtype = int)
    nums_label[0] = 0
    nums_label[1] = 0

    for i in range(np_Length):
        nums_list[i,0] = i
        nums_list[i,1] = nums_label[i]
        if nums_label[i]==1 :
            temp = i+i
            while temp<np_Length :
                nums_label[temp] = 0
                temp += i
    # print(nums_list)

    # return nums_list
    return [ nums_list[:,0], nums_list[:,1] ]

def num2_32bit(num):
    numbit = np.zeros([32], dtype = bool)
    for i in range(32):
        numbit[i] = num%2
        num /= 2
    return numbit

def prime_32bit(np_Length):
    [ x, y ] = prime_gener(np_Length)
    xbit = np.zeros([np_Length,32], dtype = bool)
    for i in range(np_Length):
        xbit[i,:] = num2_32bit(x[i])
    return [ xbit, y ]