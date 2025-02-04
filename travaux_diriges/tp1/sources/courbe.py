import matplotlib.pyplot as plt

# Données
omp_num = [1, 2, 3, 4, 5, 6, 7, 8]

speedup_2048 = [1.000, 1.797, 2.629, 3.367, 4.362, 4.764, 4.148, 4.559]
speedup_512 = [1.000, 1.860, 2.299, 3.329, 3.300, 2.963, 3.002, 3.812]
speedup_4096 = [1.000, 1.874, 2.666, 3.465, 4.125, 4.858, 4.300, 4.663]

# Tracé des courbes
plt.figure(figsize=(8, 5))
plt.plot(omp_num, speedup_2048, marker='o', linestyle='-', label='n=2048')
plt.plot(omp_num, speedup_512, marker='s', linestyle='--', label='n=512')
plt.plot(omp_num, speedup_4096, marker='^', linestyle='-.', label='n=4096')

# Labels et titre
plt.xlabel("OMP_NUM (Nombre de threads)")
plt.ylabel("Speedup")
plt.title("Speedup en fonction du nombre de threads")
plt.legend()
plt.grid(True)

# Affichage
plt.show()
