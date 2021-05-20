from numba import cuda
import numpy

def host_func(val):
    print(val)

@cuda.jit
def normalize_sq_seq(i_array, o_array):
    block = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    pos = thread + block*block_dim
    if pos < len(o_array):
        o_array[pos] = i_array[pos][0] * i_array[pos][0] + i_array[pos][1] * i_array[pos][1] + i_array[pos][2] * i_array[pos][2]
        #o_array[pos] = i_array[pos]
    #cuda.syncthreads()

@cuda.jit
def euclidean_distance(i_array_1, i_array_2, o_array):
    block = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    pos = thread + block*block_dim
    if pos < len(o_array):
        x = i_array_1[pos][0] - i_array_2[pos][0]
        y = i_array_1[pos][1] - i_array_2[pos][1]
        z = i_array_1[pos][2] - i_array_2[pos][2]
        o_array[pos] = x * x + y * y + z * z

    #cuda.syncthreads()

@cuda.jit
def euclidean_distance_from_point(i_array_1, point, o_array):
    block = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    pos = thread + block*block_dim
    if pos < len(o_array):
        x = i_array_1[pos][0] - point[0]
        y = i_array_1[pos][1] - point[1]
        z = i_array_1[pos][2] - point[2]
        o_array[pos] = x * x + y * y + z * z

    #cuda.syncthreads()

def print_array(i_array):
    i = 0
    for e in i_array:
        print("{0}: {1}".format(i,e))
        i+=1

if __name__ == "__main__":
    size = 10000000
    print(cuda.gpus)
    data = numpy.tile(numpy.array([1,2,3]),(size,1))
    data2 = numpy.tile(numpy.array([2,4,8]),(size,1))
    output = numpy.zeros(size)
    print("len(data) = {}".format(len(data)))
    print("data.shape = {}".format(data.shape))
    print("output.shape = {}".format(output.shape))
    threads_per_block = 512
    blocks_per_grid = (data.shape[0] + (threads_per_block - 1)) // threads_per_block
    print("threads_per_block: " + str(threads_per_block))
    print("blocks_per_grid: " + str(blocks_per_grid))
    print("threads_per_block * blocks_per_grid: " + str(threads_per_block*blocks_per_grid))

    #normalize_sq_seq[blocks_per_grid, threads_per_block](data, output)
    #euclidean_distance[blocks_per_grid, threads_per_block](data, data2, output)
    euclidean_distance_from_point[blocks_per_grid, threads_per_block](data, numpy.array([2,4,8]), output)
    
    print("data.shape = {}".format(data.shape))
    print("output.shape = {}".format(output.shape))
    print_array(output)
