#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/11/2 AM2:41
# @Author  : Shiloh Leung
# @Site    : 
# @File    : myTOp.py
# @Software: PyCharm Community Edition
"""
My own tensor operation module myTOp

For more mathematical details,
please refer to Tensor Decompositions and Applications by Tamara G. Kolda and Brett W. Bader

All of n-D arguments are numpy.dnarray (n>=1)
Functions:
    t2vec(T): Vectorize a tensor T into a column vector
    tnorm(T): Return the norm of tensor T
    mKron(A, B): Return the Kronecker product of matrices A and B
    mKhaR(A, B): Return the Khatri-Rao product of matrices A and B
    mHadamard(A, B): Return the Hadamard product of matrices A and B
    t2mat(T, rdim): Return the matrix transformed from tensor T in mode-rdim,
                   same as tenmat(T, rdims, 'fc') in MATLAB
"""
import numpy as np
import itertools as itl


def t2vec(T):
    # Vectorize a tensor T into a column vector
    """
    Args:
      T: Tensor, n-dimension data, created with numpy.array
    Returns:
      vec: A column vector, whose type is numpy.array, including all elements
           in tensor T
    """
    tsize = T.size
    vec = T.reshape((1, tsize))[0].transpose()
    return vec




def tnorm(T):
    # Return the norm of tensor T
    """
    Args:
      T: Tensor, n-dimension data, created with numpy.array
    Returns:
      norm: The norm of tensor T, a scalar (float)
    """
    vec = t2vec(T)
    norm = np.sqrt(sum(vec*vec))
    return norm




class NotMatError(Exception):
    pass




def mKron(A, B):
    # Return the Kronecker product of matrices A and B
    """
    Args:
      A: np.mat with shape (I, J)
      B: np.mat with shape (K, L)
    Returns:
      prod: The Kronecker product of matrices A and B,
           np.mat with shape (IK, JL)
    """
    if  len(list(A.shape))>2 or len(list(B.shape))>2:
        raise NotMatError("Both of A and B should be matrix or vector (type of np.mat)")
    I, J = A.shape
    K, L = B.shape
    repB = np.tile(B, (I, J))
    newr = I*K;  newc = J*L
    for r in range(newr):
        r_A = int(np.floor(r/K))
        for c in range(newc):
            c_A = int(np.floor(c/L))
            repB[r, c] = repB[r, c]*A[r_A, c_A]
    prod = repB
    return prod




class ColumnMatchError(Exception):
    pass




def mKhaR(A, B):
    # Return the Khatri-Rao product of matrices A and B
    """
    Args:
      A: np.mat with shape (I, K)
      B: np.mat with shape (J, K)
    Returns:
      prod: The Khatri-Rao product of matrices A and B,
           np.mat with shape (IJ, K)
    """
    if len(list(A.shape))>2 or len(list(B.shape))>2:
        raise NotMatError("Both of A and B should be matrix or vector (type of np.mat)")
    I, K_A = A.shape
    J, K_B = B.shape

    if K_A!=K_B:
        raise ColumnMatchError("Matrices A and B should have same column number")
    K = K_A
    if K==1:
        prod = mKron(A, B)
    else:
        prod = mKron(A[:, 0], B[:, 0])
        for col in range(K-1):
            A_Col = A[:, col + 1].reshape((I, 1))
            B_Col = B[:, col + 1].reshape((J, 1))
            adCol = mKron(A_Col, B_Col)
            prod = np.column_stack((prod, adCol))

    return  prod




class SizeMatchError(Exception):
    pass




def mHadamard(A, B):
    # Return the Hadamard product of matrices A and B
    """
    Args:
      A: np.mat with shape (I, J)
      B: np.mat with shape (I, J)
    Returns:
      prod: The Hadamard product of matrices A and B,
           np.mat with shape (I, J)
    """
    if A.shape != B.shape:
        raise SizeMatchError("Matrices A and B should have same size")
    I, J = A.shape
    prod = np.mat(np.zeros([I, J]))
    for r in range(I):
        for c in range(J):
            prod[r,c] = A[r,c] * B[r,c]
    return prod




class OutOfDimsError(Exception):
    pass



def t2mat(T, rdim):
    # Return the matrix transformed from tensor T in mode-rdim
    # For more mathematical details, please refer to Tensor Decompositions and Applications by Tamara G. Kolda and Brett W. Bader
    """
    Args:
      T: np.array with shape (I_1, I_2, ...., I_N)
      rdim: integer, indicating the rdim_th mode of tensor T (1<= rdim <= N)
    Returns:
      tenmat: The matrix unfolding from tensor T,
             np.mat with shape (I_rdim, I_1*...*I_{rdim-1}*I_{rdim+1}*I_{rdim+2}*...*I_N)
    """
    dims = list(T.shape)
    tdim = T.ndim    # tdim = N
    if rdim > tdim:
        raise OutOfDimsError("rdim is out of dimensions of tensor T")
    elif rdim <= 0:
        raise OutOfDimsError("rdim should be a positive integer")
    tsize = T.size
    vec = T.reshape((1, tsize))[0].transpose()    # vectorize tensor T
    idxAll = itl.product(*[range(i) for i in dims])
    idxList = []
    for i in idxAll:
        idxList.append(np.array(list(i)))
    for idxRow in idxList:
        idxRow += np.ones(tdim, dtype=int)     # make all indices >= 1

    c = int(T.size/dims[rdim-1])    # number of columns of tenmat. Obviously, col = I_{rdim+1}*I_{rdim+2}*...*I_N*I_1*...*I_{rdim-1}
    tenmat = np.mat(np.zeros( (dims[rdim-1], c) ))    # create a same size np.mat
    Jseq = np.ones(tdim, dtype=int)
    Iseq = np.array(dims)[:tdim-1]
    Jseq[1:] = Iseq
    Jseq = np.cumprod(Jseq)
    Jseq[rdim:] = Jseq[rdim:]/dims[rdim-1]    # Jseq: J_1, J_2, ..., J_rdim, J_{rdim+1}, ..., J_N
    for i in range(tsize):
        elem = vec[i]    # get corresponding element of indices idx from tensor
        idx = idxList[i]
        row = idx[rdim-1] - 1
        tmpIdx = idx -  np.ones(tdim, dtype=int)
        tmpIdx[rdim-1] = 0
        col = np.cumsum(tmpIdx*Jseq)[tdim-1]
        tenmat[row, col] = elem
    return  tenmat
