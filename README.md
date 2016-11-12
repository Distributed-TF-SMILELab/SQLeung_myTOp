# SQLeung_myTOp
Implementation of some basic tensor operations and matrix operations in Python 3.x


### Reference
1. [Tensor Decompositions and Applications](http://www.maths.manchester.ac.uk/~mlotz/teaching/nur/tensordecompositions.pdf)
2. [MATLAB Tensor Classes for FastAlgorithm Prototyping](http://www.sandia.gov/~tgkolda/pubs/pubfiles/SAND2004-5187.pdf)


### Including Operations
* The norm of a tensor
* The inner product of two same-sized tensors
* Matricization of the tensor(transforming a tensor into a matrix)
* Vectorization of the tensor(transforming a tensor into a vector)
* The product of a tensor with a scalar
* The n-mode product of a tensor with a matrix
* The n-mode product of a tensor with a vector
* Matrix Kronecker product
* Matrix Khatri-Rao product
* Matrix Hadamard product


### File Explaination
* myTOp_test.py : run it to show how to use module myTOp
* myTOp.py : put it under your current path, and 
             ```python
             import myTOp
             ``` 
             before using it
