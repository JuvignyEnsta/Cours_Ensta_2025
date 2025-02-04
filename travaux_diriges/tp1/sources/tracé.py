import matplotlib.pyplot as plt

# Données
omp_num = [1, 2, 3, 4, 5, 6, 7, 8]

speedup_1024 = [1.000, 1.860, 2.560, 3.290, 3.021, 3.947, 4.024, 3.002]
speedup_2048 = [1.000, 1.797, 2.629, 3.367, 4.362, 4.764, 4.148, 4.559]
speedup_512 = [1.000, 1.860, 2.299, 3.329, 3.300, 2.963, 3.002, 3.812]
speedup_4096 = [1.000, 1.874, 2.666, 3.465, 4.125, 4.858, 4.300, 4.663]

# Tracer les courbes
plt.figure(figsize=(10, 6))
plt.plot(omp_num, speedup_1024, marker='o', label='n = 1024')
plt.plot(omp_num, speedup_2048, marker='s', label='n = 2048')
plt.plot(omp_num, speedup_512, marker='^', label='n = 512')
plt.plot(omp_num, speedup_4096, marker='d', label='n = 4096')

# Ajouter des titres et des labels
plt.title('Speedup en fonction du nombre de threads (OMP_NUM)', fontsize=14)
plt.xlabel('Nombre de threads (OMP_NUM)', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.xticks(omp_num)  # Afficher toutes les valeurs de OMP_NUM sur l'axe x
plt.grid(True, linestyle='--', alpha=0.6)  # Ajouter une grille

# Légende
plt.legend(title='Taille du problème (n)', fontsize=10, title_fontsize=12)

# Afficher le graphique
plt.tight_layout()
plt.show()