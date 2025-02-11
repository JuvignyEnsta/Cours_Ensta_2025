# TD n° 2 - 27 Janvier 2025

##  1. Parallélisation ensemble de Mandelbrot

L'ensensemble de Mandebrot est un ensemble fractal inventé par Benoit Mandelbrot permettant d'étudier la convergence ou la rapidité de divergence dans le plan complexe de la suite récursive suivante :
$$
\left\{
\begin{array}{l}
    c\,\,\textrm{valeurs\,\,complexe\,\,donnée}\\
    z_{0} = 0 \\
    z_{n+1} = z_{n}^{2} + c
\end{array}
\right.
$$
dépendant du paramètre $c$.

Il est facile de montrer que si il existe un $N$ tel que $\mid z_{N} \mid > 2$, alors la suite $z_{n}$ diverge. Cette propriété est très utile pour arrêter le calcul de la suite puisqu'on aura détecter que la suite a divergé. La rapidité de divergence est le plus petit $N$ trouvé pour la suite tel que $\mid z_{N} \mid > 2$.

On fixe un nombre d'itérations maximal $N_{\textrm{max}}$. Si jusqu'à cette itération, aucune valeur de $z_{N}$ ne dépasse en module 2, on considère que la suite converge.

L'ensemble de Mandelbrot sur le plan complexe est l'ensemble des valeurs de $c$ pour lesquels la suite converge.

Pour l'affichage de cette suite, on calcule une image de $W\times H$ pixels telle qu'à chaque pixel $(p_{i},p_{j})$, de l'espace image, on associe une valeur complexe  $c = x_{min} + p_{i}.\frac{x_{\textrm{max}}-x_{\textrm{min}}}{W} + i.\left(y_{\textrm{min}} + p_{j}.\frac{y_{\textrm{max}}-y_{\textrm{min}}}{H}\right)$. Pour chacune des valeurs $c$ associées à chaque pixel, on teste si la suite converge ou diverge.

- Si la suite converge, on affiche le pixel correspondant en noir
- Si la suite diverge, on affiche le pixel avec une couleur correspondant à la rapidité de divergence.

1. À partir du code séquentiel `mandelbrot.py`, faire une partition équitable par bloc suivant les lignes de l'image pour distribuer le calcul sur `nbp` processus  puis rassembler l'image sur le processus zéro pour la sauvegarder. Calculer le temps d'exécution pour différents nombre de tâches et calculer le speedup. Comment interpréter les résultats obtenus ?

```
# Initialisation de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # rang du processus
size = comm.Get_size() # nombre de processus

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

# Division du travail : chaque processus calcule une portion de l'image
rows_per_proc = height // size # nombre de lignes par processus
start_row = rank * rows_per_proc # première ligne traitée par le processus
end_row = height if rank == size - 1 else (rank + 1) * rows_per_proc # dernière ligne traitée par le processus

local_convergence = np.empty((width, end_row - start_row), dtype=np.double)

# Calcul de l'ensemble de Mandelbrot pour la portion de l'image assignée
start_time = time()
for y in range(start_row, end_row):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        local_convergence[x, y - start_row] = mandelbrot_set.convergence(c, smooth=True)
end_time = time()
print(f"Process {rank} finished computation in {end_time - start_time:.4f} seconds")

# Rassemblement des données sur le processus maître (rank 0)
if rank == 0:
    convergence = np.empty((width, height), dtype=np.double)
else:
    convergence = None

comm.Gather(local_convergence, convergence, root=0)

# Constitution de l'image et sauvegarde uniquement par le processus maître
if rank == 0:
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.show()

```


* Temps d'exécution pour différents nombres de tâches et calculons le speedup

    | Nombre de tâches    | Temps d'exécution(en s) | Speedup  |
    |---------------------|-------------------------|----------|
    | 1                   | 3.2837                  | 1.0      | 
    | 2                   | 1.6921                  | 1.94     |
    | 4                   | 1.2385                  | 2.65     |

* Interprétons les résultats obtenus:
    Les résultats montrent que le speedup augmente avec le nombre de processus, ce qui indique une amélioration des performances grâce à la parallélisation en mémoire distribuée. Cependant, il peut y avoir un plateau ou une légère augmentation du temps à partir d'un certain nombre de processus en raison de la surcharge de communication entre les processus.

2. Réfléchissez à une meilleur répartition statique des lignes au vu de l'ensemble obtenu sur notre exemple et mettez la en œuvre. Calculer le temps d'exécution pour différents nombre de tâches et calculer le speedup et comparez avec l'ancienne répartition. Quel problème pourrait se poser avec une telle stratégie ?

```
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3.0 / width
scaleY = 2.25 / height

# Distribution cyclique des lignes
assigned_rows = list(range(rank, height, size))
number_of_rows = len(assigned_rows)
local_convergence = np.empty((width, number_of_rows), dtype=np.double)

start_time = time()
for idx, y in enumerate(assigned_rows):
    for x in range(width):
        c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
        local_convergence[x, idx] = mandelbrot_set.convergence(c, smooth=True)
end_time = time()
print(f"Process {rank} finished computation in {end_time - start_time:.4f} seconds")

# Préparation pour Gatherv
send_counts = None
displ = None

if rank == 0:
    counts = [(height - i - 1) // size + 1 for i in range(size)]
    send_counts = [width * cnt for cnt in counts]
    displ = [0] * size
    for i in range(1, size):
        displ[i] = displ[i-1] + send_counts[i-1]
else:
    counts = []
    send_counts = []
    displ = []

local_convergence_flat = local_convergence.reshape(-1)
big_buffer = np.empty(sum(send_counts), dtype=np.double) if rank == 0 else None

comm.Gatherv(local_convergence_flat, (big_buffer, send_counts, displ, MPI.DOUBLE), root=0)

if rank == 0:
    convergence = np.empty((width, height), dtype=np.double)
    for proc in range(size):
        rows_in_proc = counts[proc]
        for k in range(rows_in_proc):
            y = proc + k * size
            if y >= height:
                break
            start_idx = displ[proc] + k * width
            convergence[:, y] = big_buffer[start_idx: start_idx + width]

    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T) * 255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin - deb}")
    image.show()
```
* Temps d'exécution pour différents nombres de tâches et calculons le speedup

    | Nombre de tâches    | Temps d'exécution(en s) | Speedup  |
    |---------------------|-------------------------|----------|
    | 1                   | 3.2696                  | 1.0      | 
    | 2                   | 1.6012                  | 2.0419   |
    | 4                   | 0.8823                  | 3.7057   |

* Interprétons les résultats obtenus:
    Les résultats montrent que le speedup augmente avec le nombre de processus, ce qui indique une amélioration des performances grâce à la parallélisation en mémoire distribuée.


* Problème que pourrait poser une telle stratégie:
Les deux stratégies de parallélisation diffèrent dans la manière dont elles répartissent les lignes de l'image entre les processus :

- **Stratégie 1 : Répartition par blocs contigus**
   - Chaque processus reçoit un bloc de lignes consécutives à traiter.
   - Cela assure un bon accès mémoire, car les lignes sont traitées séquentiellement.
   - Un déséquilibre de charge peut survenir si le nombre de processus ne divise pas exactement la hauteur de l'image.

- **Stratégie 2 : Répartition cyclique (Round-Robin)**
   - Chaque processus traite une ligne sur `size`, assurant une distribution plus uniforme du travail.
   - Peut mieux équilibrer la charge de calcul, surtout si certaines zones de l'image sont plus complexes à calculer.
   - Risque d'un accès mémoire moins efficace à cause d'un accès dispersé aux lignes, ce qui peut nuire aux performances.

Ainsi, problème que pourrait poser une telle stratégie:

- **Moins bonne localité mémoire** : Comme les lignes sont distribuées de manière non contiguë, chaque processus doit accéder à des emplacements mémoire plus éloignés, ce qui peut ralentir les accès et affecter les performances globales.
- **Augmentation du coût de communication** : La collecte des résultats (avec `Gatherv`) peut être plus coûteuse car les données sont plus fragmentées.
- **Impact sur la mise en cache** : Dans la stratégie 1, les processeurs peuvent tirer parti de la localité spatiale des données, tandis que dans la stratégie 2, l'accès plus dispersé peut nuire à l'efficacité du cache.




3. Mettre en œuvre une stratégie maître-esclave pour distribuer les différentes lignes de l'image à calculer. Calculer le speedup avec cette approche et comparez  avec les solutions différentes. Qu'en concluez-vous ?

```
# Paramètres de l'image
width, height = 1024, 1024
scaleX, scaleY = 3.0 / width, 2.25 / height
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)

def worker(task_queue, result_queue):
    """
    Fonction exécutée par chaque esclave. 
    Prend une ligne, la calcule et envoie le résultat.
    """
    while True:
        y = task_queue.get()
        if y is None:
            break  # Fin du travail
        row = np.empty(width, dtype=np.double)
        for x in range(width):
            c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
            row[x] = mandelbrot_set.convergence(c, smooth=True)
        result_queue.put((y, row))

def main(nbp):
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Création des processus esclaves
    processes = [multiprocessing.Process(target=worker, args=(task_queue, result_queue)) for _ in range(nbp)]
    
    for p in processes:
        p.start()

    # Envoi des tâches
    start_calc = time()
    for y in range(height):
        task_queue.put(y)

    # Ajout des messages de fin
    for _ in range(nbp):
        task_queue.put(None)

    # Collecte des résultats
    convergence = np.empty((width, height), dtype=np.double)
    for _ in range(height):
        y, row = result_queue.get()
        convergence[:, y] = row
    end_calc = time()

    print(f"Temps du calcul (Maître-Esclave) : {end_calc - start_calc:.3f} s")

    # Création de l'image
    start_img = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T) * 255))
    end_img = time()
    
    print(f"Temps de constitution de l'image : {end_img - start_img:.3f} s")
    image.show()

    # Fermeture des processus
    for p in processes:
        p.join()

if __name__ == "__main__":
    nbp = min(multiprocessing.cpu_count(), 8)  # Ne pas dépasser 8 cœurs pour limiter l'overhead
    main(nbp)
```
* Calculons le speedup avec cette approche et comparons  avec les solutions différentes.

    | Nombre de tâches    | Temps d'exécution(en s) | Speedup  |
    |---------------------|-------------------------|----------|
    | 1                   | 3.118                   | 1.0      | 
    | 2                   | 3.284                   | 0.95     |
    | 4                   | 2.601                   | 1.198    |
On remarque que le speedup est beaucoup plus petit que dans les autres implémentations

* Concluons

Ce code est plus optimisé car:
- Moins de surcharge de communication: Les travailleurs récupèrent les tâches dynamiquement, évitant ainsi les périodes d’inactivité.
- Gestion propre des processus: Chaque processus est bien fermé proprement (join() après None envoyé).
- eilleur équilibre de charge: Si certaines lignes sont plus lentes, elles sont réparties dynamiquement.

Le speedup est neanmoins plus faible qu'avant car: le maître était un goulot d’étranglement
- Trop de temps perdu à envoyer/récupérer des tâches.
Si les tâches n’étaient pas bien équilibrées: Certaines lignes sont plus dures à calculer.



## 2. Produit matrice-vecteur

On considère le produit d'une matrice carrée $A$ de dimension $N$ par un vecteur $u$ de même dimension dans $\mathbb{R}$. La matrice est constituée des cœfficients définis par $A_{ij} = (i+j) \mod N$. 

Par soucis de simplification, on supposera $N$ divisible par le nombre de tâches `nbp` exécutées.

### a - Produit parallèle matrice-vecteur par colonne

Afin de paralléliser le produit matrice–vecteur, on décide dans un premier temps de partitionner la matrice par un découpage par bloc de colonnes. Chaque tâche contiendra $N_{\textrm{loc}}$ colonnes de la matrice. 

- Calculons en fonction du nombre de tâches la valeur de Nloc

Si la matrice A a une dimension globale dim (donc dim colonnes) et que l’on répartit les colonnes en blocs égaux sur P tâches (ou processus), alors le nombre de colonnes affectées à chaque tâche est
$$
N_{\text{loc}} = \frac{\text{dim}}{P}
$$

**Remarque :** Il faut que dim soit divisible par P (sinon il faudra gérer le cas où certaines tâches reçoivent un bloc de taille légèrement différente).


- Paralléliser le code séquentiel `matvec.py` en veillant à ce que chaque tâche n’assemble que la partie de la matrice utile à sa somme partielle du produit matrice-vecteur. On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.

Nous souhaitons que chaque tâche n’assemble que la partie de la matrice dont elle a besoin pour calculer sa contribution locale au produit $ v = A \cdot u $. Comme la matrice est partitionnée par blocs de colonnes, chaque processus $ p $ (avec $ p = 0, 1, \ldots, P-1 $) possède les colonnes $ j $ telles que :

$$
j \in \{ p \cdot N_{\text{loc}}, p \cdot N_{\text{loc}} + 1, \ldots, (p + 1) \cdot N_{\text{loc}} - 1 \}
$$

Pour le produit matrice-vecteur, on écrit :

$$
v_i = \sum_{j=0}^{\text{dim}-1} A_{ij} \, u_j
$$

et puisque chaque tâche possède seulement une partie des colonnes, la contribution locale de la tâche $ p $ est :

$$
v_i^{(p)} = \sum_{j=p \cdot N_{\text{loc}}}^{(p+1) \cdot N_{\text{loc}} - 1} A_{ij} \, u_j
$$

Ensuite, une opération de réduction (par exemple, un Allreduce avec l’opération somme) permettra de sommer les contributions de toutes les tâches afin que, à la fin, chaque tâche possède le vecteur $ v $ complet.


- Calculer le speed-up obtenu avec une telle approche:

Le speedup est défini par :

$$
\text{Speedup} = \frac{\text{Temps séquentiel}}{\text{Temps parallèle (global)}}
$$

Le temps parallèle global est choisi comme le maximum des temps mesurés sur tous les processus.


```
//CODE: matvec.py
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Paramètres du problème ---
dim = 1024             # Dimension de la matrice (dim x dim)
Nloc = dim // size    # Nombre de colonnes par processus (on suppose dim divisible par size)

# --- Détermination de la portion locale ---
start_col = rank * Nloc
end_col = (rank + 1) * Nloc

# Construction de la partie locale de la matrice A.
# La matrice complète est définie par : A[i, j] = ((i+j) % dim) + 1.
# Chaque processus construit uniquement les colonnes de start_col à end_col-1 pour toutes les lignes.
rows = np.arange(dim).reshape(dim, 1)
cols = np.arange(start_col, end_col).reshape(1, Nloc)
A_local = ((rows + cols) % dim) + 1.0

# Construction de la portion locale du vecteur u.
# u[j] = j + 1 pour j = start_col,...,end_col-1.
u_local = np.arange(start_col + 1, end_col + 1, dtype=float)

# --- Calcul du produit matrice-vecteur en parallèle ---
# Chaque processus calcule sa contribution locale :
#   v_local[i] = somme_{j=start_col}^{end_col-1} A[i,j] * u[j]
t_start = MPI.Wtime()
v_local = np.dot(A_local, u_local)
# La réduction (somme) de toutes les contributions locales permet d'obtenir le vecteur résultat complet.
comm.Allreduce(MPI.IN_PLACE, v_local, op=MPI.SUM)
t_end = MPI.Wtime()
temps_parallel = t_end - t_start

# On récupère le temps maximum parmi tous les processus pour représenter le temps global de l'opération parallèle.
temps_parallel_global = comm.reduce(temps_parallel, op=MPI.MAX, root=0)

# --- Calcul en séquentiel et évaluation du speed-up (réalisé sur le processus 0) ---
if rank == 0:
    # Construction complète de la matrice A et du vecteur u
    A = np.array([[(i + j) % dim + 1. for i in range(dim)] for j in range(dim)])
    u = np.array([i + 1. for i in range(dim)])
    
    # Mesure du temps de calcul séquentiel
    t_seq_start = time.time()
    v_seq = A.dot(u)
    t_seq_end = time.time()
    temps_sequentiel = t_seq_end - t_seq_start

    # Calcul du speedup
    speedup = temps_sequentiel / temps_parallel_global if temps_parallel_global > 0 else float('inf')

    # Affichage des résultats
    print("=== Résultats ===")
    print("\nVersion séquentielle :")
    print("v =", v_seq)
    print("Temps séquentiel : {:.6f} s".format(temps_sequentiel))
    
    print("\nVersion parallèle (réduite sur chaque processus) :")
    print("v =", v_local)
    print("Temps parallèle (max sur les processus) : {:.6f} s".format(temps_parallel_global))
    
    print("\nSpeedup : {:.10f}".format(speedup))

```

![Texte alternatif](./image1.png)

### b - Produit parallèle matrice-vecteur par ligne

Afin de paralléliser le produit matrice–vecteur, on décide dans un deuxième temps de partitionner la matrice par un découpage par bloc de lignes. Chaque tâche contiendra $N_{\textrm{loc}}$ lignes de la matrice.

- Calculer en fonction du nombre de tâches la valeur de Nloc

La matrice est de taille \( \text{dim} \times \text{dim} \) et chaque processus reçoit un bloc de lignes. On a donc 

$$
N_{\text{loc}} = \frac{\text{dim}}{P}
$$

où \( P \) est le nombre de tâches (processus).

Pour le processus de rang \( p \), les lignes considérées seront les indices allant de 

$$
\text{start\_row} = p \times N_{\text{loc}}
$$

à 

$$
\text{end\_row} = (p + 1) \times N_{\text{loc}} - 1.
$$


- paralléliser le code séquentiel `matvec.py` en veillant à ce que chaque tâche n’assemble que la partie de la matrice utile à son produit matrice-vecteur partiel. On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.

Chaque processus construit uniquement sa portion de la matrice \( A \) correspondant aux lignes qui lui sont attribuées et utilise le vecteur \( u \) complet (puisque la multiplication par ligne nécessite toutes les colonnes). Le produit local est 

$$
v_{\text{local}} = A_{\text{local}} \cdot u,
$$

où \( A_{\text{local}} \) est la partie de \( A \) (de taille \( N_{\text{loc}} \times \text{dim} \)) possédée par le processus.

Pour que chaque processus dispose du vecteur résultat complet \( v \), on réalise un Allgather qui rassemble tous les \( v_{\text{local}} \) dans un vecteur \( v \) de taille \( \text{dim} \).

- Calculer le speed-up obtenu avec une telle approche

Le temps de calcul parallèle est mesuré (avec `MPI_Wtime`) et, sur le processus 0, on effectue également le produit matrice–vecteur en séquentiel. Le speed-up est alors calculé comme

$$
\text{Speedup} = \frac{\text{Temps séquentiel}}{\text{Temps parallèle (global)}}
$$

Le temps parallèle global est ici choisi comme le maximum des temps locaux mesurés sur l'ensemble des processus.

```
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Paramètres du problème ---
dim = 1024             # Dimension de la matrice (dim x dim)
Nloc = dim // size     # Nombre de lignes par processus (on suppose que dim est divisible par size)

# --- Détermination de la portion locale de la matrice ---
# Pour le processus de rang 'rank', les lignes vont de start_row à end_row-1
start_row = rank * Nloc
end_row = (rank + 1) * Nloc

# Construction de la portion locale de la matrice A.
# La matrice complète est définie par : A[i, j] = ((i+j) % dim) + 1.
# Pour le bloc de lignes du processus, i va de start_row à end_row-1 et j de 0 à dim-1.
rows = np.arange(start_row, end_row).reshape(Nloc, 1)  # dimensions (Nloc, 1)
cols = np.arange(dim).reshape(1, dim)                  # dimensions (1, dim)
A_local = ((rows + cols) % dim) + 1.0

# Construction du vecteur u (commun à tous les processus)
# u[j] = j + 1 pour j = 0, 1, ..., dim-1
u = np.array([j + 1. for j in range(dim)], dtype=float)

# --- Calcul du produit matrice-vecteur en parallèle ---
# Chaque processus calcule le produit pour ses lignes locales :
#   v_local[i] = somme_{j=0}^{dim-1} A[i,j] * u[j]
t_start = MPI.Wtime()
v_local = np.dot(A_local, u)  # v_local a pour taille (Nloc,)
t_local = MPI.Wtime() - t_start

# Rassemblement des résultats partiels afin d'obtenir le vecteur v complet sur tous les processus.
# On utilise MPI.Allgather pour que chaque processus reçoive l'ensemble des contributions.
v = np.empty(dim, dtype=float)
comm.Allgather([v_local, MPI.DOUBLE], [v, MPI.DOUBLE])

# Récupération du temps maximum parmi tous les processus pour représenter le temps global de l'opération parallèle.
t_global = comm.reduce(t_local, op=MPI.MAX, root=0)

# --- Calcul en séquentiel et évaluation du speed-up (réalisé sur le processus 0) ---
if rank == 0:
    # Construction complète de la matrice A et du vecteur u
    A_full = np.array([[(i + j) % dim + 1. for i in range(dim)] for j in range(dim)])
    u_full = np.array([i + 1. for i in range(dim)], dtype=float)
    
    # Mesure du temps de calcul séquentiel
    t_seq_start = time.time()
    v_seq = A_full.dot(u_full)
    t_seq_end = time.time()
    t_seq = t_seq_end - t_seq_start
    
    # Calcul du speedup
    speedup = t_seq / t_global if t_global > 0 else float('inf')
    
    # Affichage des résultats
    print("=== Produit matrice-vecteur par partitionnement par lignes ===")
    print("\nRésultat séquentiel :")
    print("v =", v_seq)
    print("Temps séquentiel : {:.6f} s".format(t_seq))
    
    print("\nRésultat parallèle :")
    print("v =", v)
    print("Temps parallèle (max sur les processus) : {:.6f} s".format(t_global))
    
    print("\nSpeedup obtenu : {:.2f}".format(speedup))

```
![](./image1.png)

## 3. Entraînement pour l'examen écrit

Alice a parallélisé en partie un code sur une machine à mémoire distribuée. Pour un jeu de données spécifique, elle remarque que la partie exécutée en parallèle représente **90 %** du temps d’exécution du programme en séquentiel.

En utilisant la **loi d’Amdahl**, pouvez-vous prédire l’accélération maximale que pourra obtenir Alice avec son code (en considérant $n \gg 1$) ?

À votre avis, pour ce jeu de données spécifique, quel nombre de nœuds de calcul semble-t-il raisonnable de prendre pour ne pas trop gaspiller de ressources CPU ?

En effectuant son calcul sur son calculateur, Alice s’aperçoit qu’elle obtient une accélération maximale de **4** en augmentant le nombre de nœuds de calcul pour son jeu de données.

En doublant la quantité de données à traiter et en supposant une complexité linéaire de l’algorithme parallèle, quelle accélération maximale peut espérer Alice en utilisant la **loi de Gustafson** ?

---

## Correction – Accélération Maximale en Parallélisme

Soit un programme dont **90 %** du temps d'exécution est parallélisable (fraction parallèle $p = 0,9$) et **10 %** est séquentiel (fraction sérielle $s = 0,1$).

### 1. Avec la loi d’Amdahl

La vitesse d’exécution maximale théorique (pour $n \to \infty$) est donnée par :

$$
S_{\max} = \frac{1}{s} = \frac{1}{0,1} = 10.
$$

**Interprétation :**  
Même avec un nombre infini de nœuds, l’accélération maximale du programme est **10**, en raison de la partie séquentielle incompressible.

### 2. Choix du nombre de nœuds

En pratique, Alice observe un **speedup maximal de 4**.  
**Conclusion :**  
Il est raisonnable d’utiliser environ **4 nœuds** pour ce jeu de données, car au-delà, le gain devient marginal et le gaspillage de ressources CPU augmente.

### 3. Avec la loi de Gustafson

Si on **double** la quantité de données à traiter, on suppose que :
- Le **temps séquentiel** reste constant ($T_s$),
- Le **temps parallèle** double (passant de $T_p$ à $2T_p$).

Pour le problème initial (temps total $T = T_s + T_p$), nous avions :
- $T_s = 0,1T$,
- $T_p = 0,9T$.

Après doublement des données :
- Temps total $T' = T_s + 2T_p = 0,1T + 1,8T = 1,9T$.

La nouvelle fraction sérielle devient :

$$
\alpha = \frac{T_s}{T'} = \frac{0,1T}{1,9T} \approx 0,0526.
$$

La loi de Gustafson donne le speedup sur $n$ nœuds :

$$
S_G(n) = \alpha + n \times (1 - \alpha) \approx 0,0526 + 0,9474 \, n.
$$

**Exemples :**
- Pour **$n = 4$ nœuds** :  
$$ 
S_G(4) \approx 0,0526 + 0,9474 \times 4 \approx 3,84 
$$  
(similaire à l’observation précédente).
- Pour **$n = 10$ nœuds** :  
$$ 
S_G(10) \approx 0,0526 + 0,9474 \times 10 \approx 9,53.
$$

### Conclusion

En doublant la quantité de données et en supposant une complexité linéaire dans la partie parallèle, la **loi de Gustafson** prédit un speedup qui **augmente presque linéairement avec le nombre de nœuds**. Ainsi, pour un grand nombre de nœuds ($n \gg 1$), le gain théorique tend vers :

$$
S_G(n) \approx 0,95 \, n.
$$

En pratique, cela signifie qu’avec suffisamment de ressources, l’accélération peut **presque** atteindre le nombre de nœuds utilisés, bien que des limitations matérielles et d’autres facteurs puissent ralentir ce gain théorique.
