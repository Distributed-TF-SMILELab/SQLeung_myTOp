#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/11/2 AM2:41
# @Author  : Shiloh Leung
# @Site    : 
# @File    : myTOp.py
# @Software: PyCharm Community Edition
"""
My own tensor operation module myTOp
All of n-D arguments are numpy.dnarray (n>=1)
Functions:
    t2vec(T): Vectorize a tensor T into a column vector
    tnorm(T): Return the norm of tensor T
    mKron(A, B): Return the Kronecker product of matrices A and B
    mKhaR(A, B): Return the Khatri-Rao product of matrices A and B
    mHadamard(A, B): Return the Hadamard product of matrices A and B
"""
import numpy as np



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
