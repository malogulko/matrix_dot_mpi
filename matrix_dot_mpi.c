//
// Created by Malogulko, Alexey on 21/04/2020.
//

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.c"

// Shows which row of the grid this node is on
int grid_row(int partition_rank, int partitions_width) {
    return (int) (partition_rank/partitions_width);
}

int a_offset(int partition_rank, int partitions_width, int partition_size) {
    return grid_row(partition_rank, partitions_width) * partition_size;
}

// Shows which col of the grid this node is on
int grid_col(int partition_rank, int partitions_width) {
    return partition_rank%partitions_width;
}

int b_offset(int partition_rank, int partitions_width, int partition_size) {
    return grid_col(partition_rank, partitions_width)  * partition_size;
}

/**
 * Classic IJK matrix multiplication
 * @param matrix_a - row-wise addressed matrix_a
 * @param matrix_b - column-wise addressed matrix_b
 * @param matrix_c - row-wise addressed matrix_c
 * @param matrix_size - size of the matrix
 */
 /// There must be an ability for non-square matrices
void ijk(double *matrix_a, double *matrix_b, double *matrix_c, int matrix_size, int partition_size) {
    for (int i = 0; i < partition_size; i++) {
        for (int j = 0; j < partition_size; j++) {
            double *c_sum = matrix_c + i * partition_size + j;
            for (int k = 0; k < matrix_size; k++) {
                *(c_sum) += *(matrix_a + i * matrix_size + k) * *(matrix_b + j * matrix_size + k);
            }
        }
    }
}

int main(int argc, char **argv) {
    int num_partitions, matrix_size, partition_rank;
    parse_matrix_size(argc, argv, &matrix_size);
    double start, end;

    // Initializes MPI here
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_partitions);
    MPI_Comm_rank(MPI_COMM_WORLD, &partition_rank);

    // Making sure partitioning is possible
    check_partition(matrix_size, num_partitions);

    // Full length of the stripe of memory for partition
    int partition_size = (matrix_size * matrix_size) / num_partitions;
    // This number represents how many partitions there are in row/col
    int partitions_width = (int) sqrt(num_partitions);

    //
    // This describes concept of "Local Square", e.g. because there must be 1:1 square-to-node mapping,
    // we're going to split these matrices into local squares, such as:
    //
    // #############
    // # sq0 # sq1 #
    // #############
    // # sq2 # sq3 #
    // #############
    //
    // Where sqX - node number(rank).
    // However, because for multiplication you'll need full segment of two squares, e.g.
    // to calculate full sq0 of matrix C, the node will need sq0 and sq2 form matrix A and sq0 and sq1 from matrix B
    // We're going to introduce concept of "Local stripe", which is a full block required for multiplication

    // Local offset shows where THIS partition-generated data is located INSIDE local stripe for matrix A
    // e.g. if you're at node 2 in 4 node cluster, you have A stripe starting at 1 * 2(because node 2 is at 1st row)
    // if at node 1, A stripe starting at 0 * 2(because node 1 is at 0 row). Where 2 is number of partitions in row/col

    int local_offset_a = a_offset(partition_rank, partitions_width, partition_size); // at which row this node is
    int local_offset_b = b_offset(partition_rank, partitions_width, partition_size); // at which col this node is

    int grid_row_pos = grid_row(partition_rank, partitions_width);
    int grid_col_pos = grid_col(partition_rank, partitions_width);

    printf("Partition %d of %d total, partition size %d, matrix %d x %d partitions, place in grid %dx%d, local offset A %d, B %d\n",
           partition_rank, num_partitions, partition_size, partitions_width, partitions_width, grid_row_pos, grid_col_pos, local_offset_a, local_offset_b);

    double *a_local_stripe = malloc(partition_size * partitions_width * sizeof(double));
    double *b_local_stripe = malloc(partition_size * partitions_width * sizeof(double));
    double *c_local_block = malloc(partition_size * sizeof(double));
    random_matrix(a_local_stripe + local_offset_a, partition_size);
    random_matrix(b_local_stripe + local_offset_b, partition_size);


    // Blocks until all processes in the communicator have reached this routine
    MPI_Barrier(MPI_COMM_WORLD);
    if (partition_rank == 0)
        printf("Passed first barrier\n");
        start = MPI_Wtime();

    // Here we're syncing a_local_stripe
    for (int dst_rank = 0; dst_rank < num_partitions; dst_rank += partitions_width) {
        // Here we send our a partition node with dst_rank to complete their a_local_stripe, as well as completing ours
        if (dst_rank != partition_rank && grid_col_pos == grid_col(dst_rank, partitions_width)) { // only other in same column
            int remote_offset = a_offset(dst_rank, partitions_width, partition_size);
            printf("Syncing a_local_stripe, local offset %d, remote %d from node %d to %d\n", local_offset_a, remote_offset, partition_rank, dst_rank);
            MPI_Gather(a_local_stripe + local_offset_a,
                       partition_size, MPI_DOUBLE,
                       a_local_stripe + remote_offset,
                       partition_size, MPI_DOUBLE, dst_rank, MPI_COMM_WORLD
            );
        }
    }

    if (partition_rank == 0)
        printf("a_local_stripe sync completed\n");

    // Blocking sync
    MPI_Barrier(MPI_COMM_WORLD);

    // Here we're syncing b_local_stripe
    for (int dst_rank = 0; dst_rank < num_partitions; dst_rank += partitions_width) {
        // Here we send our a partition node with dst_rank to complete their a_local_stripe, as well as completing ours
        if (dst_rank != partition_rank && grid_row_pos == grid_row(dst_rank, partitions_width)) { // only other nodes
            int remote_offset = b_offset(dst_rank, partitions_width, partition_size);
            printf("Syncing b_local_stripe, local offset %d, remote %d from node %d to %d\n", local_offset_b, remote_offset, partition_rank, dst_rank);
            MPI_Gather(b_local_stripe + local_offset_b,
                       partition_size, MPI_DOUBLE,
                       b_local_stripe + remote_offset,
                       partition_size, MPI_DOUBLE, dst_rank, MPI_COMM_WORLD
            );
        }
    }

    if (partition_rank == 0)
        printf("b_local_stripe sync completed\n");
    // Blocking sync
    MPI_Barrier(MPI_COMM_WORLD);

    // Now we have both stripes completed, it's time to multiply
    ijk(a_local_stripe, b_local_stripe, c_local_block, matrix_size, (int) sqrt(partition_size));

    // Once calculations are complete, we can free stripes
    free(a_local_stripe);
    free(b_local_stripe);
    MPI_Barrier(MPI_COMM_WORLD);

    double *matrix_c;

    if (partition_rank == 0) {
        matrix_c = malloc(matrix_size * matrix_size * sizeof(double));
    }

    MPI_Gather(c_local_block, partition_size, MPI_DOUBLE, matrix_c, partition_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (partition_rank == 0) {
        end = MPI_Wtime();
        printf("Matrix C memory stripe:\n");
        for (int l = 0; l < matrix_size * matrix_size; ++l) {
            printf("%f ", *(matrix_c + l));
        }
        printf("\n");
        free(matrix_c);
    }

    MPI_Finalize();
}