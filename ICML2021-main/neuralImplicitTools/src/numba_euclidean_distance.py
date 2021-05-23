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
def min_between_arrays_store_in_first(io_array_1, i_array_2):
    block = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    pos = thread + block*block_dim
    if pos < len(io_array_1):
        if io_array_1[pos] > i_array_2[pos]:
            io_array_1[pos] = i_array_2[pos]
    #cuda.syncthreads()

def pytho_call_min_between_arrays_store_in_first(io_array_1, i_array_2, threads_per_block=1024):
    if not (io_array_1.shape[0] == i_array_2.shape[0]):
        print("io_array_1.shape[0] must equal i_array_2.shape[0]")
        return
    
    blocks_per_grid = (io_array_1.shape[0] + (threads_per_block - 1)) // threads_per_block
    min_between_arrays_store_in_first[blocks_per_grid, threads_per_block](io_array_1, i_array_2)

@cuda.jit
def min_between_arrays_store_in_first_2(io_array_1, i_array_2):
    block = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    pos = thread + block*block_dim
    if pos < len(io_array_1):
        cuda.atomic.min(io_array_1,pos,i_array_2[pos])
    
    #cuda.syncthreads()

def pytho_call_min_between_arrays_store_in_first_2(io_array_1, i_array_2, threads_per_block=1024):
    if not (io_array_1.shape[0] == i_array_2.shape[0]):
        print("io_array_1.shape[0] must equal i_array_2.shape[0]")
        return
    
    blocks_per_grid = (io_array_1.shape[0] + (threads_per_block - 1)) // threads_per_block
    min_between_arrays_store_in_first[blocks_per_grid, threads_per_block](io_array_1, i_array_2)

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

def pytho_call_euclidean_distance_from_point(point, array, output, threads_per_block=1024):
    if not (array.shape[0] == output.shape[0]):
        print("array.shape[0] must equal output.shape[0]")
        return
    # TODO: chequear que el tamaño de array[0] y point sea igual a 3

    blocks_per_grid = (array.shape[0] + (threads_per_block - 1)) // threads_per_block
    euclidean_distance_from_point[blocks_per_grid, threads_per_block](array, point, output)

@cuda.jit
def euclidean_distance_from_point_variation(i_array_1, point, normals_array_1, normal_point, o_array):
    block = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    pos = thread + block*block_dim
    if pos < len(o_array):
        x = i_array_1[pos][0] - point[0]
        y = i_array_1[pos][1] - point[1]
        z = i_array_1[pos][2] - point[2]
        #cosAngle = normal_point.dot(normals_array_1[pos])
        cosAngle = normals_array_1[pos][0]*normal_point[0] + normals_array_1[pos][1]*normal_point[1] + normals_array_1[pos][2]*normal_point[2]
        o_array[pos] = (x * x + y * y + z * z) / (cosAngle + 0.0001)

    #cuda.syncthreads()

def pytho_call_euclidean_distance_from_point_variation(point, array, pointNormal, arrayNormals, output, threads_per_block=1024):
    if not (array.shape[0] == output.shape[0]):
        print("array.shape[0] must equal output.shape[0]")
        return
    # TODO: chequear que el tamaño de array[0] y point sea igual a 3

    blocks_per_grid = (array.shape[0] + (threads_per_block - 1)) // threads_per_block
    euclidean_distance_from_point_variation[blocks_per_grid, threads_per_block](array, point, arrayNormals, pointNormal, output)


@cuda.jit
def numba_arg_max(array, originalIndices, outputIndices, mid, end):
    block = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    pos = thread + block*block_dim
    if pos < mid:
        if (pos+mid) < end:
            if array[originalIndices[pos]] < array[originalIndices[pos+mid]]:
                outputIndices[pos] = originalIndices[pos+mid]
            else:
                outputIndices[pos] = originalIndices[pos]
        else:
            outputIndices[pos] = originalIndices[pos]

def argmax(array, originalIndices, tmpIndices, threads_per_block=1024):
    if not (array.shape[0] == originalIndices.shape[0]):
        print("array.shape[0] must equal originalIndices.shape[0]")
        return
    
    end = originalIndices.shape[0]
    if end == 0:
        print("must have at least one element")
        return
    
    mid = (end + 1) // 2     #techo de division
    
    if tmpIndices.shape[0] < mid:
        print("tmpIndices.shape[0] must be greater than or equal to ceiling of originalIndices.shape[0] / 2")
        return
    
    if end == 1:
        return originalIndices[0]
    else:
        blocks_per_grid = (mid + (threads_per_block - 1)) // threads_per_block
        numba_arg_max[blocks_per_grid, threads_per_block](array, originalIndices, tmpIndices, mid, end)
        end = mid
        mid = (end + 1) // 2     #techo de division
    
    while end > 1:
        blocks_per_grid = (mid + (threads_per_block - 1)) // threads_per_block
        numba_arg_max[blocks_per_grid, threads_per_block](array, tmpIndices, tmpIndices, mid, end)
        end = mid
        mid = (end + 1) // 2     #techo de division
    
    return tmpIndices[0]

@cuda.jit
def numba_arg_max_2(array, tmpMaxValue, outMaxIndex):
    block = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    pos = thread + block*block_dim
    if pos < len(array):
        cuda.atomic.max(tmpMaxValue,0,array[pos])
        cuda.syncthreads()
        if array[pos]==tmpMaxValue[0]:
            outMaxIndex[0] = pos
    
    cuda.syncthreads()

def argmax2(array, threads_per_block=1024):
    blocks_per_grid = (array.shape[0] + (threads_per_block - 1)) // threads_per_block
    
    outIndex = numpy.array([-1])
    tmpMaxVal = numpy.array([array[0]])
    numba_arg_max_2[blocks_per_grid, threads_per_block](array, tmpMaxVal, outIndex)

    return outIndex[0]

def print_array(i_array):
    i = 0
    for e in i_array:
        print("{0}: {1}".format(i,e))
        i+=1

if __name__ == "__main__":
    size = 10000000
    print(cuda.gpus)
    data = numpy.tile(numpy.array([1,2,3]),(size,1))
    #data = numpy.array([4,0,1,2])
    #data2 = numpy.tile(numpy.array([2,4,8]),(size,1))
    #data2 = numpy.arange(size) * 2
    data2 = numpy.arange(size, dtype=numpy.uint64)
    
    data3 = numpy.arange(size)

    output = numpy.zeros(size)
    #tmpIndices = numpy.zeros(size, dtype=numpy.uint64)
    tmpIndices = numpy.zeros((size + 1) // 2, dtype=numpy.uint64)

    #print("len(data) = {}".format(len(data)))
    print("data.shape = {}".format(data.shape))
    print("data[2] = {}".format(data[2]))
    print("data2.shape = {}".format(data2.shape))
    print("data2[2] = {}".format(data2[2]))
    print("output.shape = {}".format(output.shape))
    print("tmpIndices.shape = {}".format(tmpIndices.shape))
    #threads_per_block = 512
    #blocks_per_grid = (data.shape[0] + (threads_per_block - 1)) // threads_per_block
    #print("threads_per_block: " + str(threads_per_block))
    #print("blocks_per_grid: " + str(blocks_per_grid))
    #print("threads_per_block * blocks_per_grid: " + str(threads_per_block*blocks_per_grid))

    #normalize_sq_seq[blocks_per_grid, threads_per_block](data, output)
    #euclidean_distance[blocks_per_grid, threads_per_block](data, data2, output)
    #pytho_call_euclidean_distance_from_point(numpy.array([2,4,8]), data, output)
    #min_between_arrays_store_in_first[blocks_per_grid, threads_per_block](data, data2)
    #maxIdx = argmax(data3, data2, tmpIndices)
    #pytho_call_min_between_arrays_store_in_first(data2, data3)
    
    print("data.shape = {}".format(data.shape))
    print("data2.shape = {}".format(data2.shape))
    #print_array(data)
    #print("maxIdx = {0}, maxVal = {1}".format(maxIdx, data3[maxIdx]))
    maxIdx_2 = argmax2(data3)
    print("maxIdx_2 = {0}, maxVal_2 = {1}".format(maxIdx_2, data3[maxIdx_2]))
