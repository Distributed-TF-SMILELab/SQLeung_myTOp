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
print("\nusing t2vec to vetorize the tensor T : \n    vec = t2vec(T)")
vec = mto.t2vec(T)
print("show vec :\n", vec)
print("shape of vector vec : ", vec.shape, '\n')

# calculate norm of tensor
print("\n"*2, '='*12, "calculate norm of tensor", '='*12, "\n")
T1 = np.array([[1,2,3], [4,5,6]])
print("show tensor T1 : \n", T1)
normT1 = mto.tnorm(T1)
print("\nusing tnorm to calculate norm of tensor T1 : \n    normT1 = tnorm(T1)")
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
print("\nusing mKron to calculate Kronecker product of A1 and B1 : \n    prodKron = mKron(A1, B1)\n")
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
print("\nusing mKhaR to calculate Khatri-Rao product of A2 and B2 : \n    prodKhaR = mKhaR(A2, B2)\n")
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
print("\nusing mHadamard to calculate Hadamard product of A3 and B3 : \n    prodHada = mHadamard(A3, B3)\n")
print("shape of prodHada : ", prodHada.shape)
print("prodHada = \n", prodHada)


print('='*50)