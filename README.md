## What?

This is an example of parallel matrix multiplication algorithm based on MPI. In this example, every node generates it's 
own piece of matrix A and B and then sends it to other nodes involved in the computation.

## Example

Imaging you want to calculate dot product for 16x16 square matrix on 4 nodes. Your matrix will be partitioned like:

```
┌---┬---┐
│ 0 │ 1 │
├---┼---┤
│ 2 | 3 │ 
└---┴---┘
```

Where every square is a 4x4 block, and every 0 block of matrices A and B is randomly filled by node 0.
However, to calculate the block 0 of matrix C, you'l need full stripes for both matrix, e.g. 

```

Matrix A stripe:
┌---┐
│ 0 │
├---┤
│ 2 │ 
└---┘

Matrix B stripe:
┌---┬---┐
│ 0 │ 1 │
└---┴---┘

```

In this case, the following communications occurs:


* Matrix A: Node 0 sends block 0 to node 2, Node 2 sends block 2 to node 0 - both nodes have full column segment 
* Matrix B: Node 0 sends block 0 to node 1, Node 1 sends block 1 to node 0 - both nodes have full row segment

Then, once all required data is gathered, every node will start calculation, produces it's own block matrix C and then 
transfers them to node 0, which assembles full matrix C from received blocks.

## Building

Linux:
```shell script
.build# mpicc -o matrix_dot_mpi ../matrix_dot_mpi.c -lm
```

cmake on Mac:
```shell script
.build# cmake ..
.build# make
```

## Running

```
                   ┌---------------------- Number of nodes
                   |                   ┌-- X size of square matrix
.build# mpirun -np 4 ./matrix_dot_mpi 16
16;4;3;19
 | | |  |
 | | |  └--- Time spent in calculation and data transfer(microseconds)
 | | └------ Time spent in data transfer only(microseconds)
 | └-------- Number of nodes
 └---------- X size of square matrix
```