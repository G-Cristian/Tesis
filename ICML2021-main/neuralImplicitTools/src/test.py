import numpy as np 

#constants used for sampling box AND miniball normalization
BOUNDING_SPHERE_RADIUS = 0.9
SAMPLE_SPHERE_RADIUS = 1.0

def mirrorByX(array):
    mirror = lambda point: [-point[0],point[1],point[2]]

    return np.apply_along_axis(mirror,1,array)

def mirrorByY(array):
    mirror = lambda point: [point[0],-point[1],point[2]]
    
    return np.apply_along_axis(mirror,1,array)

def mirrorByZ(array):
    mirror = lambda point: [point[0],point[1],-point[2]]
    
    return np.apply_along_axis(mirror,1,array)

def mirrorByXY(array):
    mirror = lambda point: [-point[0],-point[1],point[2]]

    return np.apply_along_axis(mirror,1,array)

def mirrorByXZ(array):
    mirror = lambda point: [-point[0],point[1],-point[2]]
    
    return np.apply_along_axis(mirror,1,array)

def mirrorByYZ(array):
    mirror = lambda point: [point[0],-point[1],-point[2]]

    return np.apply_along_axis(mirror,1,array)

def mirrorByXYZ(array):
    mirror = lambda point: [-point[0],-point[1],-point[2]]

    return np.apply_along_axis(mirror,1,array)

def mirrorByPlanes(array, planes):
    if planes == 'x':
        return mirrorByX(array)
    elif planes == 'y':
        return mirrorByY(array)
    elif planes == 'z':
        return mirrorByZ(array)
    elif planes == 'xy':
        tmp = mirrorByX(array)
        tmp = np.concatenate((tmp,mirrorByY(array)))
        tmp = np.concatenate((tmp,mirrorByXY(array)))
        return tmp
    elif planes == 'xz':
        tmp = mirrorByX(array)
        tmp = np.concatenate((tmp,mirrorByZ(array)))
        tmp = np.concatenate((tmp,mirrorByXZ(array)))
    elif planes == 'yz':
        tmp = mirrorByY(array)
        tmp = np.concatenate((tmp,mirrorByZ(array)))
        tmp = np.concatenate((tmp,mirrorByYZ(array)))
    elif planes == 'xyz':
        tmp = mirrorByX(array)
        tmp = np.concatenate((tmp,mirrorByY(array)))
        tmp = np.concatenate((tmp,mirrorByZ(array)))
        tmp = np.concatenate((tmp,mirrorByXY(array)))
        tmp = np.concatenate((tmp,mirrorByXZ(array)))
        tmp = np.concatenate((tmp,mirrorByYZ(array)))
        tmp = np.concatenate((tmp,mirrorByXYZ(array)))

        return tmp
    
    return None

if __name__ == '__main__':
    array = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    print("array.shape = {}".format(array.shape))
    print("array = {}".format(array))

    X = mirrorByX(array)
    print("X.shape = {}".format(X.shape))
    print("X = {}".format(X))

    concatenatedMirrorByXYZ = mirrorByPlanes(array, "xyz")
    print("concatenatedMirrorByXYZ.shape = {}".format(concatenatedMirrorByXYZ.shape))
    print("concatenatedMirrorByXYZ = {}".format(concatenatedMirrorByXYZ))

    #total_1 = np.concatenate((array,mirrorByPlanes(array, "")))
    #print("total_1.shape = {}".format(total_1.shape))
    #print("total_1 = {}".format(total_1))

    total_2 = np.concatenate((array,mirrorByPlanes(array, "xyz")))
    print("total_2.shape = {}".format(total_2.shape))
    print("total_2 = {}".format(total_2))
