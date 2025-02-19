from mpi4py import MPI
import numpy as np
from PIL import Image
import matplotlib.cm
from math import log
import time

class MandelbrotSet:
    def __init__(self, max_iterations=50, escape_radius=10):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def convergence(self, c: complex, smooth=True) -> float:
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

# 1. **初始化 MPI**
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # 进程号
nbp = comm.Get_size()   # 总进程数

# 2. **定义图像参数**
width, height = 1024, 1024
scaleX = 3.0 / width
scaleY = 2.25 / height

# 3. **计算进程应该处理的行**
rows_per_proc = height // nbp  # 每个进程计算的行数
start_row = rank * rows_per_proc
end_row = (rank + 1) * rows_per_proc if rank != nbp - 1 else height

# 4. **计算 Mandelbrot**
mandelbrot_set = MandelbrotSet(max_iterations=100, escape_radius=10)
local_convergence = np.empty((rows_per_proc, width), dtype=np.float64)

start_time = time.time()
for i, y in enumerate(range(start_row, end_row)):
    for x in range(width):
        c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
        local_convergence[i, x] = mandelbrot_set.convergence(c, smooth=True)
end_time = time.time()
local_time = end_time - start_time

# 5. **主进程收集数据**
if rank == 0:
    global_convergence = np.empty((height, width), dtype=np.float64)
else:
    global_convergence = None

comm.Gather(local_convergence, global_convergence, root=0)

# 6. **主进程保存图像**
if rank == 0:
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(global_convergence) * 255))
    image.save("mandelbrot_mpi.png")
    print(f"Temps d'exécution (nbp={nbp}) : {local_time:.4f} s")
