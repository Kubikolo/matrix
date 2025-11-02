import numpy as np
import tracemalloc
import time
import csv

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

def pad_to_even(M):
    n = len(M)
    if n % 2 == 0:
        return M, False
    else:
        return np.pad(M, ((0, 1), (0, 1))), True

def binet(A, B, ops):
    if len(A) == 1:
        ops[0] += 1
        return np.dot(A, B)

    A, flag_A = pad_to_even(A)
    B, flag_B = pad_to_even(B)

    flag = flag_A | flag_B

    n = len(A)
    mid = n//2

    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    C11 = binet(A11, B11, ops) + binet(A12, B21, ops)
    C12 = binet(A11, B12, ops) + binet(A12, B22, ops)
    C21 = binet(A21, B11, ops) + binet(A22, B21, ops)
    C22 = binet(A21, B12, ops) + binet(A22, B22, ops)
    ops[0] += 8*mid*mid

    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    if flag:
        C = C[:-1, :-1]

    return C

# def binet_slow(A, B, ops):
#     n = len(A)

#     if n == 1:
#         ops[0] += 1
#         return np.dot(A, B)

    
#     mid = n//2

#     A11 = A[:mid, :mid]
#     A12 = A[:mid, mid:]
#     A21 = A[mid:, :mid]
#     A22 = A[mid:, mid:]
#     B11 = B[:mid, :mid]
#     B12 = B[:mid, mid:]
#     B21 = B[mid:, :mid]
#     B22 = B[mid:, mid:]

#     C11 = binet(A11, B11, ops) + binet(A12, B21, ops)
#     C12 = binet(A11, B12, ops) + binet(A12, B22, ops)
#     C21 = binet(A21, B11, ops) + binet(A22, B21, ops)
#     C22 = binet(A21, B12, ops) + binet(A22, B22, ops)
#     ops[0] += 8*mid*mid

#     C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

#     return C

def strassen(A, B, ops):
    if len(A) == 1:
        ops[0] += 1
        return np.dot(A, B)

    A, flag_A = pad_to_even(A)
    B, flag_B = pad_to_even(B)

    flag = flag_A | flag_B

    n = len(A)
    mid = n//2

    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    M1 = strassen(A11, B12 - B22, ops)
    M2 = strassen(A11 + A12, B22, ops)
    M3 = strassen(A21 + A22, B11, ops)
    M4 = strassen(A22, B21 - B11, ops)
    M5 = strassen(A11 + A22, B11 + B22, ops)
    M6 = strassen(A12 - A22, B21 + B22, ops)
    M7 = strassen(A11 - A21, B11 + B12, ops)

    C11 = M5 + M4 - M2 + M6
    C12 = M1 + M2
    C21 = M3 + M4
    C22 = M5 + M1 - M3 - M7

    ops[0] += 18*mid*mid

    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    if flag:
        C = C[:-1, :-1]

    return C

binet_data = []
strassen_data = []

for i in range(2, 11):
    print("ITERACJA:", i)

    A, B = generate_multipliable_matrices(i, seed=1)

    # ==== BINET ====

    ops = [0]

    tracemalloc.start()
    s = time.perf_counter()
    binet(A, B, ops)
    binet_time = time.perf_counter() - s
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    binet_mem = peak
    binet_ops = ops[0]
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


    # ==== STRASSEN ====

    ops = [0]

    tracemalloc.start()
    s = time.perf_counter()
    strassen(A, B, ops)
    strassen_time = time.perf_counter() - s
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    strassen_mem = peak
    strassen_ops = ops[0]
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

    # ==== BINET SLOW ====

    # tracemalloc.start()
    # s = time.perf_counter()
    # power_two = 2**int(np.ceil(np.log2(np.max(A.shape))))
    # A_pad = np.pad(A, ((0, power_two-A.shape[0]), (0, power_two-A.shape[1])))
    # B_pad = np.pad(B, ((0, power_two-B.shape[0]), (0, power_two-B.shape[1])))
    # binet_slow(A_pad, B_pad, ops)
    # strassen_time = time.perf_counter() - s
    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # strassen_mem = peak
    # strassen_ops = ops[0]
    # print("CZAS:", strassen_time)
    # print("OPERACJE:", strassen_ops)
    # print("PAMIEC:", strassen_mem)
    # print()

    # strassen_data.append({
    #     "size": i,
    #     "time_sec": strassen_time,
    #     "ops": strassen_ops,
    #     "memory_bytes": strassen_mem
    # })

with open("binet_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["size", "time_sec", "ops", "memory_bytes"])
    writer.writeheader()
    writer.writerows(binet_data)

with open("strassen_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["size", "time_sec", "ops", "memory_bytes"])
    writer.writeheader()
    writer.writerows(strassen_data)