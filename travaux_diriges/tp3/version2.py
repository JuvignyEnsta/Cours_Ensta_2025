from mpi4py import MPI
import numpy as np

def pprint(*args, **kwargs):
    """Ajoute l'ID du processus aux prints pour le débogage"""
    print(f'[{MPI.COMM_WORLD.rank}]', *args, **kwargs)

def bucket_sort_v2(N=100000):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = MPI.Wtime()  

    np.random.seed(rank)
    nb_data_loc = N // size
    data_loc = np.random.rand(nb_data_loc)
    data_loc.sort()

    # **1. 采样数据以计算全局分位数**
    sample_size = max(1, nb_data_loc // 10)
    local_sample = data_loc[::sample_size]
    all_samples = comm.gather(local_sample, root=0)

    if rank == 0:
        all_samples = np.concatenate(all_samples)
        global_quantiles = np.percentile(all_samples, np.linspace(0, 100, size + 1))
    else:
        global_quantiles = None

    global_quantiles = comm.bcast(global_quantiles, root=0)

    # **2. 分配数据到正确的桶**
    send_data = [[] for _ in range(size)]
    for val in data_loc:
        for i in range(size):
            if global_quantiles[i] <= val < global_quantiles[i + 1] or (i == size - 1 and val == global_quantiles[i + 1]):
                send_data[i].append(val)
                break

    send_data = [np.array(bucket, dtype=float) for bucket in send_data]

    # **计算未发送的数据**
    all_sent_data = np.concatenate(send_data) if len(send_data) > 0 else np.array([])
    local_remaining_data = np.setdiff1d(data_loc, all_sent_data, assume_unique=True)

    # **打印交换前的数据个数**
    pprint("avant:", len(data_loc))

    # **3. 计算发送/接收数据大小**
    send_counts = np.array([len(bucket) for bucket in send_data], dtype=int)
    recv_counts = np.zeros(size, dtype=int)
    comm.Alltoall(send_counts, recv_counts)

    # **计算全局 send_counts 和 recv_counts**
    total_send_counts = np.sum(send_counts)
    total_recv_counts = np.sum(recv_counts)
    global_send_counts = comm.allreduce(total_send_counts, op=MPI.SUM)
    global_recv_counts = comm.allreduce(total_recv_counts, op=MPI.SUM)

    # **确保所有进程总的 send_counts == recv_counts**
    assert global_send_counts == global_recv_counts, f"ERREUR: global_send_counts {global_send_counts} != global_recv_counts {global_recv_counts}"

    # **打印 send_counts 和 recv_counts 的总数**
    pprint("send_counts sum:", total_send_counts)
    pprint("recv_counts sum:", total_recv_counts)

    send_data_flat = np.concatenate(send_data) if np.sum(send_counts) > 0 else np.array([], dtype=float)
    recv_data_flat = np.empty(sum(recv_counts), dtype=float)

    send_displs = np.insert(np.cumsum(send_counts), 0, 0)[:-1]
    recv_displs = np.insert(np.cumsum(recv_counts), 0, 0)[:-1]

    # **4. 使用 MPI_Alltoallv 交换数据**
    comm.Alltoallv([send_data_flat, send_counts, send_displs, MPI.DOUBLE],
                   [recv_data_flat, recv_counts, recv_displs, MPI.DOUBLE])

    recv_data = np.split(recv_data_flat, recv_displs[1:]) if np.sum(recv_counts) > 0 else []

    # **5. 处理接收到的数据并排序**
    data_sorted = np.concatenate([*recv_data, local_remaining_data]) if len(recv_data) > 0 else local_remaining_data
    data_sorted.sort()

    # **打印交换后的数据个数**
    pprint("apres:", len(data_sorted))

    total_time = MPI.Wtime() - start_time  

    # **6. 进程 0 输出最终排序结果**
    if rank == 0:
        print(f"\n=== Résultats du tri parallèle (Version 2) ===")
        print(f"Temps total : {total_time:.6f} secondes")

if __name__ == "__main__":
    bucket_sort_v2(N=100000)
