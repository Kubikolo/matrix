import numpy as np
import tracemalloc

class CustomMatrix:
    def __init__(self, matrix, power_two):
        self.matrix = matrix
        self.power_two = 2**int(np.ceil(np.log2(np.max(matrix.shape)))) if power_two is None else power_two
    
    def shape(self):
        return self.matrix.shape
    
    def split(self):
        half = self.power_two//2
        return CustomMatrix(self.matrix[:half, :half], half), \
                CustomMatrix(self.matrix[:half, half:], half), \
                CustomMatrix(self.matrix[half:, :half], half), \
                CustomMatrix(self.matrix[half:, half:], half)

    @staticmethod
    def combine(M11, M12, M21, M22):
        # wszystkie Mkn mają takie samo power_two
        if M11.shape()[0]*M11.shape()[1] == 0:
            return CustomMatrix(M11.matrix, M11.power_two*2)
        if M12.shape()[0]*M12.shape()[1] == 0:
            if M21.shape()[0]*M21.shape()[1] == 0:
                return CustomMatrix(M11.matrix, M11.power_two*2)
            return CustomMatrix(np.block([[M11.matrix], [M21.matrix]]), M11.power_two*2)
        if M21.shape()[0]*M21.shape()[1] == 0:
            return CustomMatrix(np.block([[M11.matrix, M12.matrix]]), M11.power_two*2)
        
        # hacky, ale inaczej strassen nie działa
        # musi tak być bo w strassenie w M22 w najpłytszej rekursji, większość sie zeruje
        # więc albo trzeba de-padding, albo zpaddingować reszte
        # > looks inside
        M12.matrix = np.pad(M12.matrix, ((0, 0), (0, M22.shape()[1]-M12.shape()[1])))
        M21.matrix = np.pad(M21.matrix, ((0, M22.shape()[0]-M21.shape()[0]), (0, 0)))
        return CustomMatrix(np.block([[M11.matrix, M12.matrix], 
                     [M21.matrix, M22.matrix]]), M11.power_two*2)
    
    def __add__(self, other):
        max_shape = np.maximum(self.matrix.shape, other.matrix.shape)
        global ops
        ops += max_shape[0]*max_shape[1]
        padded_first = np.pad(self.matrix, ((0, max_shape[0]-self.matrix.shape[0]), (0, max_shape[1]-self.matrix.shape[1])))
        padded_second = np.pad(other.matrix, ((0, max_shape[0]-other.matrix.shape[0]), (0, max_shape[1]-other.matrix.shape[1])))
        return CustomMatrix(padded_first + padded_second, self.power_two)

    def __sub__(self, other):
        max_shape = np.maximum(self.matrix.shape, other.matrix.shape)
        global ops
        ops += max_shape[0]*max_shape[1]
        padded_first = np.pad(self.matrix, ((0, max_shape[0]-self.matrix.shape[0]), (0, max_shape[1]-self.matrix.shape[1])))
        padded_second = np.pad(other.matrix, ((0, max_shape[0]-other.matrix.shape[0]), (0, max_shape[1]-other.matrix.shape[1])))
        return CustomMatrix(padded_first - padded_second, self.power_two)

    @staticmethod
    def multiply_binet(A, B):
        if A.power_two == 1 and B.power_two == 1:
            if A.matrix.shape[0]*A.matrix.shape[1]*B.matrix.shape[0]*B.matrix.shape[1] == 0:
                return CustomMatrix(np.array([[]]), 1)
            
            global ops
            ops += 1
            return CustomMatrix(np.array([[A.matrix[0][0] * B.matrix[0][0]]]), 1)
        
        A11, A12, A21, A22 = A.split()
        B11, B12, B21, B22 = B.split()

        C11 = CustomMatrix.multiply_binet(A11, B11) + CustomMatrix.multiply_binet(A12, B21)
        C12 = CustomMatrix.multiply_binet(A11, B12) + CustomMatrix.multiply_binet(A12, B22)
        C21 = CustomMatrix.multiply_binet(A21, B11) + CustomMatrix.multiply_binet(A22, B21)
        C22 = CustomMatrix.multiply_binet(A21, B12) + CustomMatrix.multiply_binet(A22, B22)

        return CustomMatrix.combine(C11, C12, C21, C22)
    
    @staticmethod
    def multiply_strassen(A, B):
        if A.power_two == 1 and B.power_two == 1:
            if A.matrix.shape[0]*A.matrix.shape[1]*B.matrix.shape[0]*B.matrix.shape[1] == 0:
                return CustomMatrix(np.array([[]]), 1)
            
            global ops
            ops += 1
            return CustomMatrix(np.array([[A.matrix[0][0] * B.matrix[0][0]]]), 1)
        
        A11, A12, A21, A22 = A.split()
        B11, B12, B21, B22 = B.split()

        M1 = CustomMatrix.multiply_strassen(A11 + A22, B11 + B22)
        M2 = CustomMatrix.multiply_strassen(A21 + A22, B11)
        M3 = CustomMatrix.multiply_strassen(A11, B12 - B22)
        M4 = CustomMatrix.multiply_strassen(A22, B21 - B11)
        M5 = CustomMatrix.multiply_strassen(A11 + A12, B22)
        M6 = CustomMatrix.multiply_strassen(A21 - A11, B11 + B12)
        M7 = CustomMatrix.multiply_strassen(A12 - A22, B21 + B22)

        C11 = M1+M4-M5+M7
        C12 = M3+M5
        C21 = M2+M4
        C22 = M1-M2+M3+M6

        return CustomMatrix.combine(C11, C12, C21, C22)
    
    def __repr__(self):
        return f"{self.matrix} {self.power_two}"
    
def generate_multipliable_matrices(
    dim,
    min_size=1, max_size=10, 
    dtype=float, 
    seed=None
):

    rng = np.random.default_rng(seed)

    A = rng.integers(min_size, max_size, size=(dim, dim)).astype(dtype)
    B = rng.integers(min_size, max_size, size=(dim, dim)).astype(dtype)

    return A, B

def init_custom_matrices(A, B):
    A_power = 2**int(np.ceil(np.log2(np.max(A.shape))))
    B_power = 2**int(np.ceil(np.log2(np.max(B.shape))))
    power_two = max(A_power, B_power)
    return CustomMatrix(A, power_two), CustomMatrix(B, power_two)

import time
import csv

binet_data = []
strassen_data = []

for i in range(2, 101):
    print("ITERACJA:", i)

    ops = 0

    A, B = generate_multipliable_matrices(i, seed=1)
    A, B = init_custom_matrices(A, B)

    tracemalloc.start()
    s = time.perf_counter()
    CustomMatrix.multiply_binet(A, B)
    binet_time = time.perf_counter() - s
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    binet_mem = peak
    binet_ops = ops
    print("CZAS:", binet_time)
    print("OPERACJE:", binet_ops)
    print("PAMIEC:", binet_mem)
    print()

    binet_data.append({
        "size": i,
        "time_sec": binet_time,
        "ops": binet_ops,
        "memory_bytes": binet_mem
    })

    ops = 0

    tracemalloc.start()
    s = time.perf_counter()
    CustomMatrix.multiply_strassen(A, B)
    strassen_time = time.perf_counter() - s
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    strassen_mem = peak
    strassen_ops = ops
    print("CZAS:", strassen_time)
    print("OPERACJE:", strassen_ops)
    print("PAMIEC:", strassen_mem)
    print()

    strassen_data.append({
        "size": i,
        "time_sec": strassen_time,
        "ops": strassen_ops,
        "memory_bytes": strassen_mem
    })

with open("binet_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["size", "time_sec", "ops", "memory_bytes"])
    writer.writeheader()
    writer.writerows(binet_data)

with open("strassen_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["size", "time_sec", "ops", "memory_bytes"])
    writer.writeheader()
    writer.writerows(strassen_data)