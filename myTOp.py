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
    tinner(T1, T2): Return the inner product of tensor T1 and T2
    mKron(A, B): Return the Kronecker product of matrices A and B
    mKhaR(A, B): Return the Khatri-Rao product of matrices A and B
    mHadamard(A, B): Return the Hadamard product of matrices A and B
    t2mat(T, rdim): Return the matrix transformed from tensor T in mode-rdim
    mat2t(A, rdim, dims): Return tensor T transformed from matrix A whose rows correspond to mode-rdim of tensor T
    ttmat(T, A, rdim): Return the rdim-mode product of tensor T and matrix A
"""
import numpy as np
import itertools as itl


class NotMatError(Exception):
    pass


class SizeMatchError(Exception):
    pass


class ColumnMatchError(Exception):
    pass


class OutOfDimsError(Exception):
    pass



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
    vec = np.mat(t2vec(T)).T
    norm = (vec.H*vec)[0,0]
    return norm



def tinner(T1, T2):
    # Return the inner product of tensor T1 and T2
    """
    Args:
      T1: Tensor, n-dimension data, numpy.array with shape of (I_1, I_2, ..., I_N), over C
      T2: Tensor, n-dimension data, numpy.array with shape of (I_1, I_2, ..., I_N), over C
    Returns:
      inprod: The inner product of tensor T1 and T2, a scalar
    """
    if T1.shape != T1.shape:
        raise SizeMatchError("tensor T1 should have same shape of tensor T2")
    vec1 = np.mat(t2vec(T1)).T
    vec2 = np.mat(t2vec(T2)).T
    inprod = (vec1.H * vec2)[0,0]
    return inprod





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







def t2mat(T, rdim):
    # Return the matrix transformed from tensor T in mode-rdim
    # For more mathematical details, please refer to Tensor Decompositions and Applications by Tamara G. Kolda and Brett W. Bader
    """
    Args:
      T: np.array with shape (I_1, I_2, ...., I_N)
      rdim: integer, indicating the rdim_th mode of tensor T (1<= rdim <= N)
    Returns:
      tenmat: The matrix transformed from tensor T,
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
        row = idx[rdim-1]
        idx[rdim - 1] = 0
        col = np.cumsum(idx * Jseq)[tdim - 1]
        tenmat[row, col] = elem
    return  tenmat



def mat2t(A, rdim, dims):
    # Return tensor T transformed from matrix A whose rows correspond to mode-rdim of tensor T
    """
        Args:
          A: The matrix transformed from tensor T,
                 np.mat with shape (I_rdim, I_1*...*I_{rdim-1}*I_{rdim+1}*I_{rdim+2}*...*I_N)
          rdim: Integer, indicating matrix A is transformed from tensor T in mode rdim (1<= rdim <= N)
          dims: List of each dimension of tensor T
        Returns:
          T: The tensor transformed from matrix A, np.array with shape (I_1, I_2, ...., I_N)
    """
    tdim = len(dims)
    if rdim > tdim:
        raise OutOfDimsError("rdim is out of dimensions of tensor T")
    elif rdim <= 0:
        raise OutOfDimsError("rdim should be a positive integer")
    r, c = A.shape
    if r!=dims[rdim-1]:
        raise SizeMatchError("the row number of A should correspond to rdim in dims")
    tsize = int(r*c)
    idxAll = itl.product(*[range(i) for i in dims])
    idxList = []
    for i in idxAll:
        idxList.append(np.array(list(i)))
    Jseq = np.ones(tdim, dtype=int)
    Iseq = np.array(dims)[:tdim - 1]
    Jseq[1:] = Iseq
    Jseq = np.cumprod(Jseq)
    Jseq[rdim:] = Jseq[rdim:] / dims[rdim - 1]  # Jseq: J_1, J_2, ..., J_rdim, J_{rdim+1}, ..., J_N
    vec = np.zeros(tsize)
    for i in range(tsize):    # map matrix A into a vector according to dims
        idx = idxList[i]
        row = idx[rdim - 1]
        idx[rdim - 1] = 0
        col = np.cumsum(idx * Jseq)[tdim - 1]
        vec[i] = A[row, col]  # get corresponding element of indices idx from A

    T = vec.reshape(dims)
    return T




def ttmat(T, A, rdim):
    # Return the rdim-mode product of tensor T and matrix A
    """
    Args:
      T: np.array with shape (I_1, I_2, ..., I_{rdim-1}, I_rdim, I_{rdim+1}, ..., I_N)
      A: np.mat with shape (J, I_rdim)
      rdim: integer, indicating multiplying tensor T by matrix A in mode-rdim (1<= rdim <= N)
    Returns:
      prodT: The rdim-mode product of tensor T and matrix A,
             np.array with shape (I_1, I_2, ..., I_{rdim-1}, J, I_{rdim+1}, ..., I_N)
    """
    J, K = A.shape
    dims = list(T.shape)
    tdim = T.ndim
    if rdim > tdim:
        raise OutOfDimsError("rdim is out of dimensions of tensor T")
    elif rdim <= 0:
        raise OutOfDimsError("rdim should be a positive integer")
    if K!=dims[rdim-1]:
        raise SizeMatchError("mode-rdim of tensor T should match the column number of matrix A")
    Tmat = t2mat(T, rdim)
    prodmat = A * Tmat
    newdims = dims
    newdims[rdim-1] = J
    prodT = mat2t(prodmat, rdim, newdims)
    return  prodT