//
// Created by Malogulko, Alexey on 21/04/2020.
//

#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.c"

// Shows which row of the grid this node is on
int grid_row(int partition_rank, int partitions_width) {
    return (int) (partition_rank / partitions_width);
}

// Shows which col of the grid this node is on
int grid_col(int partition_rank, int partitions_width) {
    return partition_rank % partitions_width;
}

/**
 * Classic IJK matrix multiplication
 * @param matrix_a - row-wise addressed matrix_a
 * @param matrix_b - column-wise addressed matrix_b
 * @param matrix_c - row-wise addressed matrix_c
 * @param matrix_size - size of the full matrix
 * @param partition_size - size of the block to calculate
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
    struct timespec start, end;

    // Initializes MPI here
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_partitions);
    MPI_Comm_rank(MPI_COMM_WORLD, &partition_rank);

    // Making sure partitioning is possible
    if (partition_rank == 0)
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

    int grid_row_pos = grid_row(partition_rank, partitions_width);
    int grid_col_pos = grid_col(partition_rank, partitions_width);

    //printf("Partition %d of %d total, partition size %d, matrix %d x %d partitions, place in grid %dx%d\n",
    //       partition_rank, num_partitions, partition_size, partitions_width, partitions_width, grid_row_pos,
    //       grid_col_pos);

    // Allocating local blocks
    double *a_local_block = malloc(partition_size * sizeof(double));
    double *b_local_block = malloc(partition_size * sizeof(double));

    // Filling local blocks with randomness
    random_matrix(a_local_block, partition_size);
    random_matrix(b_local_block, partition_size);

    // Splitting global communicator into row-based blocks based on which row in grid current node is
    int mpi_col_in_row_rank, mpi_row_size; // rank in row(col), row size
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, grid_row_pos, partition_rank, &row_comm);
    MPI_Comm_rank(row_comm, &mpi_col_in_row_rank);
    MPI_Comm_size(row_comm, &mpi_row_size);

    //printf("Communication group for row %d, world rank %d, row rank %d, row size %d\n",
    //       grid_row_pos, partition_rank, mpi_col_in_row_rank, mpi_row_size);

    // Splitting global communicator  into col-based blocks based on which col in grid current node is
    int mpi_row_in_col_rank, mpi_col_size; // rank in col(row), col size
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, grid_col_pos, partition_rank, &col_comm);
    MPI_Comm_rank(col_comm, &mpi_row_in_col_rank);
    MPI_Comm_size(col_comm, &mpi_col_size);

    // printf("Communication group for col %d, world rank %d, row rank %d, row size %d\n",
    //        grid_col_pos, partition_rank, mpi_row_in_col_rank, mpi_col_size);

    // Blocks until all processes in the communicator have reached this routine
    MPI_Barrier(MPI_COMM_WORLD);

    if (partition_rank == 0)
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    //printf("Syncing a_local_col col %d, node %d(world)/%d(col)\n", grid_col_pos, partition_rank, mpi_col_in_row_rank);
    double *a_local_col = malloc(partition_size * partitions_width * sizeof(double));
    // Syncing an entire column of matrix A into a_local_col. MPI_Allgather will order received data by ranks of the
    // nodes in the column(rows), so in the end a_local_col will represent full col of matrix blocks
    MPI_Allgather(a_local_block, partition_size, MPI_DOUBLE, a_local_col, partition_size, MPI_DOUBLE, col_comm);
    free(a_local_block); // No need to have this block anymore


    //printf("Syncing b_local_row row %d, node %d(world)/%d(row)\n", grid_row_pos, partition_rank, mpi_row_in_col_rank);
    double *b_local_row = malloc(partition_size * partitions_width * sizeof(double));
    // Syncing an entire row of matrix B into b_local_row. MPI_Allgather will order received data by ranks of the
    // nodes in the row(cols), so in the end b_local_row will represent full row of matrix blocks
    MPI_Allgather(b_local_block, partition_size, MPI_DOUBLE, b_local_row, partition_size, MPI_DOUBLE, col_comm);
    free(b_local_block); // No need to have this block anymore

    double *c_local_block = malloc(partition_size * sizeof(double));

    //printf("Calculating matrix C block %dx%d\n", grid_row_pos, grid_col_pos);
    // Now we have both stripes completed, it's time to multiply
    ijk(a_local_col, b_local_row, c_local_block, matrix_size, (int) sqrt(partition_size));

    // Once calculations are complete, we can free local rows and cols
    free(a_local_col);
    free(b_local_row);
    MPI_Barrier(MPI_COMM_WORLD);

    double *matrix_c;

    if (partition_rank == 0) {
        matrix_c = malloc(matrix_size * matrix_size * sizeof(double));
    }

    // Gathering all blocks(c_local_block) of matrix_c into global matrix on node 0
    MPI_Gather(c_local_block, partition_size, MPI_DOUBLE, matrix_c, partition_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (partition_rank == 0) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        //printf("Matrix C memory stripe:\n");
        //for (int l = 0; l < matrix_size * matrix_size; ++l) {
        //    printf("%f ", *(matrix_c + l));
        //}
        //printf("\n");
        uint64_t delta_us =
                (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000; // microseconds
        printf("%d;%d;%llu\n", matrix_size, num_partitions, delta_us);
        free(matrix_c);
    }

    MPI_Finalize();
}