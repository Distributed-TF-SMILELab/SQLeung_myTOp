#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/11/6 PM3:33
# @Author  : Shiloh Leung
# @Site    : 
# @File    : myTOp_test.py
# @Software: PyCharm Community Edition

"""
test on myTOp
"""

import myTOp as mto
import numpy as np

print('='*50)
print("\n"," "*11, "test on tensor toolbox myTOp"," "*11, "\n")


# vectorization of tensor
print('='*12, "Vectorization of tensor",  "="*12, "\n")
T = np.array([[[1,2,3,4],[5,6,7,8]], [[9,10,11,12], [13,14,15,16]]])
print("shape of tensor T : ", T.shape)
print("show tensor T : \n", T)
print("\n-- using t2vec to vetorize the tensor T : \n      vec = t2vec(T)")
vec = mto.t2vec(T)
print("show vec :\n", vec)
print("shape of vector vec : ", vec.shape, '\n')




# calculate norm of tensor
print("\n"*2, '='*12, "calculate norm of tensor", '='*12, "\n")
T1 = np.array([[1,2,3], [4,5,6]])
print("show tensor T1 : \n", T1)
normT1 = mto.tnorm(T1)
print("\n-- using tnorm to calculate norm of tensor T1 : \n      normT1 = tnorm(T1)")
print("\nnormT1 = ", normT1)
print("\ncalculate in another way:")
print("    normT1_2 = np.sqrt(sum(sum(T1*T1)))")
print("\nnormT1_2 = ", np.sqrt(sum(sum(T1*T1))))




# calculate Kronecker product of 2 matrices
print("\n"*2, '='*12, "Kronecker product of 2 matrices", "="*12, "\n")
A1 = np.mat([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
B1 = np.mat([[1,1,1],[2,2,2]])
print("shape of matrix A1 : ", A1.shape)
print("show matrix A1 :\n", A1)
print("\nshape of matrix B1 : ", B1.shape)
print("show matrix B1 :\n", B1)
prodKron = mto.mKron(A1, B1)
print("\n-- using mKron to calculate Kronecker product of A1 and B1 : \n      prodKron = mKron(A1, B1)\n")
print("shape of prodKron : ", prodKron.shape)
print("prodKron = \n", prodKron)




# calculate Khatri-Rao product of 2 matrices
print("\n"*2, '='*12, "Khatri-Rao product of 2 matrices", "="*12, "\n")
A2 = np.mat([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
B2 = np.mat([[1,1,1,1],[2,2,2,2]])
print("shape of matrix A2 : ", A2.shape)
print("show matrix A2 :\n", A2)
print("\nshape of matrix B2 : ", B2.shape)
print("show matrix B2 :\n", B2)
prodKhaR = mto.mKhaR(A2, B2)
print("\n-- using mKhaR to calculate Khatri-Rao product of A2 and B2 : \n      prodKhaR = mKhaR(A2, B2)\n")
print("shape of prodKhaR : ", prodKhaR.shape)
print("prodKhaR = \n", prodKhaR)





# calculate Hadamard product of 2 matrices
print("\n"*2, '='*12, "Hadamard product of 2 matrices", "="*12, "\n")
A3 = np.mat([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
B3 = np.mat([[1,1,1,1], [2,2,2,2], [3, 3, 3, 3]])
print("shape of matrix A3 : ", A3.shape)
print("show matrix A3 :\n", A3)
print("\nshape of matrix B3 : ", B3.shape)
print("show matrix B3 :\n", B3)
prodHada = mto.mHadamard(A3, B3)
print("\n-- using mHadamard to calculate Hadamard product of A3 and B3 : \n        prodHada = mHadamard(A3, B3)\n")
print("shape of prodHada : ", prodHada.shape)
print("prodHada = \n", prodHada)




# transfer a tensor into a matrix
print("\n"*2, '='*12, "Matricization of tensor", "="*12, "\n")
T2 = np.array([[[1, 13], [4, 16], [7, 19], [10, 22]], [[2, 14], [5, 17], [8, 20], [11, 23]], [[3, 15], [6, 18], [9, 21], [12, 24]]])
print("shape of matrix T2 : ", T2.shape)
print("show tensor T2 :\n", T2)
T2mat_1 = mto.t2mat(T2, 1)
T2mat_2 = mto.t2mat(T2, 2)
T2mat_3 = mto.t2mat(T2, 3)
print("\n\n-- using t2mat to transfer the tensor T2 into matrix in mode-1: \n      T2mat_1 = t2mat(T2, 1)")
print("\nshape of matrix T2mat_1 : ", T2mat_1.shape)
print("show matrix T2mat_1 :\n", T2mat_1)
print("\n\n-- using t2mat to transfer the tensor T2 into matrix in mode-2: \n      T2mat_2 = t2mat(T2, 2)")
print("\nshape of matrix T2mat_2 : ", T2mat_2.shape)
print("show matrix T2mat_2 :\n", T2mat_2)
print("\n\n-- using t2mat to transfer the tensor T2 into matrix in mode-3: \n      T2mat_3 = t2mat(T2, 3)")
print("\nshape of matrix T2mat_3 : ", T2mat_3.shape)
print("show matrix T2mat_3 :\n", T2mat_3)

# transfer a tensor into a matrix
print("\n"*2, '='*12, "Transforming matrix back to tensor", "="*12, "\n")
dims = [3,4,2]
print("dims = ", dims)
print("\n\n-- using t2mat to transfer matrix Tmat_1 back to tensor T1  : \n      T2_1 = mat2t(T2mat_1, 1, dims)")
T2_1 = mto.mat2t(T2mat_1, 1, dims)
print("\nshape of tensor T2_1 : ", T2_1.shape)
print("show tensor T2_1 :\n", T2_1)
print("\nT2_1 == T2 : \n", T2_1==T2)
print("\n\n-- using t2mat to transfer matrix Tmat_2 back to tensor T2  : \n      T2_2 = mat2t(T2mat_2, 2, dims)")
T2_2 = mto.mat2t(T2mat_2, 2, dims)
print("\nshape of tensor T2_2 : ", T2_2.shape)
print("show tensor T2_2 :\n", T2_2)
print("\nT2_2 == T2 : \n", T2_2==T2)
print("\n\n-- using t2mat to transfer matrix Tmat_3 back to tensor T3  : \n      T2_3 = mat2t(T2mat_3, 3, dims)")
T2_3 = mto.mat2t(T2mat_3, 3, dims)
print("\nshape of tensor T2_3 : ", T2_3.shape)
print("show tensor T2_3 :\n", T2_3)
print("\nT2_3 == T2 : \n", T2_3==T2)


print('='*50)