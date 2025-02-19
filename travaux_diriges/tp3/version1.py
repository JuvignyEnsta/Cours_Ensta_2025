from mpi4py import MPI
import numpy as np

def pprint(*args, **kwargs):
    """Ajoute l'ID du processus aux prints pour le débogage"""
    print(f'[{MPI.COMM_WORLD.rank}]', *args, **kwargs)

def bucket_sort_v1(N=10):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = MPI.Wtime()

    # **1. 生成本地数据并排序**
    np.random.seed(rank)
    nb_data_loc = N // size
    data_loc = np.random.rand(nb_data_loc)
    data_loc.sort()

    # **2. 计算全局 min 和 max**
    min_global = comm.allreduce(np.min(data_loc), op=MPI.MIN)
    max_global = comm.allreduce(np.max(data_loc), op=MPI.MAX)

    # **3. 计算桶的范围**
    buckets = np.linspace(min_global, max_global, size + 1)

    # **4. 分配数据到正确的桶**
    send_data = [[] for _ in range(size)]
    for val in data_loc:
        for i in range(size):
            if buckets[i] <= val < buckets[i + 1] or (i == size - 1 and val == buckets[i + 1]):
                send_data[i].append(val)
                break

    send_data = [np.array(bucket, dtype=float) for bucket in send_data]

    # **打印交换前的数据**
    pprint('avant:', len(data_loc), data_loc)

    # **5. 计算发送/接收数据大小**
    send_counts = np.array([len(bucket) for bucket in send_data], dtype=int)
    recv_counts = np.zeros(size, dtype=int)

    # 交换数据量信息
    comm.Alltoall(send_counts, recv_counts)

    # **打印发送和接收的数据数量**
    pprint("send_counts:", send_counts, "sum:", np.sum(send_counts))
    pprint("recv_counts:", recv_counts, "sum:", np.sum(recv_counts))

    send_data_flat = np.concatenate(send_data) if np.sum(send_counts) > 0 else np.array([], dtype=float)
    recv_data_flat = np.empty(sum(recv_counts), dtype=float)

    send_displs = np.insert(np.cumsum(send_counts), 0, 0)[:-1]
    recv_displs = np.insert(np.cumsum(recv_counts), 0, 0)[:-1]

    # **6. 使用 MPI_Alltoallv 交换数据**
    comm.Alltoallv([send_data_flat, send_counts, send_displs, MPI.DOUBLE],
                   [recv_data_flat, recv_counts, recv_displs, MPI.DOUBLE])

    recv_data = np.split(recv_data_flat, recv_displs[1:]) if np.sum(recv_counts) > 0 else []

    # **7. 处理接收到的数据并排序**
    data_sorted = np.concatenate(recv_data) if len(recv_data) > 0 else np.array([])
    data_sorted.sort()

    # **打印交换后的数据**
    pprint('apres:', len(data_sorted), data_sorted)

    total_time = MPI.Wtime() - start_time

    # **8. 进程 0 输出排序结果**
    if rank == 0:
        print(f"\n=== Résultats du tri parallèle (Version 1) ===")
        print(f"Nombre total d'éléments triés : {len(data_sorted)}")
        print(f"Temps total : {total_time:.6f} secondes")
        print(f"Processus 0 : Premiers 20 éléments triés : {data_sorted[:20]}")

if __name__ == "__main__":
    bucket_sort_v1(N=10)
