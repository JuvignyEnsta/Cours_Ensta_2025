 Processeur 0  Processeur 1
1- Affichage / calcul : séparer l'affichage et le calcul  
- Séquentiel 
Calcul1 - Afficge 1 - calcul 2 - Affiche 2

- Parallélisation
Calcul1 Affiche 1     Affiche2
        Calcul 2      Calcul 3

Afiche 0    Affiche 1     
    Calcul1     Calcul2   
2- D/D : processeur 0 affiche, P1 et P2 calculent

3- DD + AC (AC= Affichage - calcul)

4- Asynchrone affiche : C
|Affiche     |
|Calcul1...10|

NB: créer des branches pour chaque question du tp
et faire des merge requests pour que le prof puisse voir, facilite la correction et permet de

MPI_Probe: est ce qu'il y'a des données en attente que je peux consommer


app.dra()

Send(true, tag= 100)

Compute()
    if (MPI_probe(msg='...', tag = 100))
        MPI_send(grid)
    else() // ici on continue à faire la génération en attendant que le mpi probe se réalise

NB: MPI probe est une fonction qui ne coûte presque rien


# 
    elif 
        if rank == 1:
        compute_next_gen(grid, comm)
else:
    comm.Recv(appli.grid.data,)
    
    #pour arrêter le programme
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            loop = False
            comm.Abort()
            print(f"Temps calcul prochaine generation : {t2-t1:2.2e} secondes, temps affichage : {t3-t2:2.2e} secondes\r", end='')
            
Question 1.2
if iter_count%100:
comm.send(grid, dest=0, tag=1)


Decomposition de domaine: La zone de cellules fantôme doit être la plus petite possible

La valeur d'une cellule depend de ses 4 voisins
grille fantome = 402*600 gauche
grille fantome = 402*600 droite
