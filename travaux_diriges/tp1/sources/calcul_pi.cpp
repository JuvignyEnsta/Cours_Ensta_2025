# include <chrono>
# include <random>
# include <cstdlib>
# include <sstream>
# include <string>
# include <fstream>
# include <iostream>
# include <iomanip>
# include <mpi.h>
# include <omp.h>

// Attention , ne marche qu'en C++ 11 ou supérieur :
double approximate_pi_sequential(unsigned long nbSamples) 
{
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = beginning.time_since_epoch();
    unsigned seed = d.count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    unsigned long nbDarts = 0;

    for (unsigned sample = 0; sample < nbSamples; ++sample) {
        double x = distribution(generator);
        double y = distribution(generator);
        // Test if the dart is in the unit disk
        if (x * x + y * y <= 1) nbDarts++;
    }

    // Number of nbDarts throwed in the unit disk
    double ratio = double(nbDarts) / double(nbSamples);
    return 4 * ratio;
}

double approximate_pi_parallel(unsigned long nbSamples) 
{
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = beginning.time_since_epoch();
    unsigned seed = d.count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    unsigned long nbDarts = 0;

    #pragma omp parallel
    {
        std::default_random_engine thread_generator(seed + omp_get_thread_num());
        unsigned long thread_nbDarts = 0;

        #pragma omp for
        for (unsigned sample = 0; sample < nbSamples; ++sample) {
            double x = distribution(thread_generator);
            double y = distribution(thread_generator);
            // Test if the dart is in the unit disk
            if (x * x + y * y <= 1) thread_nbDarts++;
        }

        #pragma omp atomic
        nbDarts += thread_nbDarts;
    }

    // Number of nbDarts throwed in the unit disk
    double ratio = double(nbDarts) / double(nbSamples);
    return 4 * ratio;
}

int main(int nargs, char* argv[])
{
    // On initialise le contexte MPI qui va s'occuper :
    //    1. Créer un communicateur global, COMM_WORLD qui permet de gérer
    //       et assurer la cohésion de l'ensemble des processus créés par MPI;
    //    2. d'attribuer à chaque processus un identifiant ( entier ) unique pour
    //       le communicateur COMM_WORLD
    //    3. etc...
    MPI_Init(&nargs, &argv);
    // Pour des raisons de portabilité qui débordent largement du cadre
    // de ce cours, on préfère toujours cloner le communicateur global
    // MPI_COMM_WORLD qui gère l'ensemble des processus lancés par MPI.
    MPI_Comm globComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &globComm);
    // On interroge le communicateur global pour connaître le nombre de processus
    // qui ont été lancés par l'utilisateur :
    int nbp;
    MPI_Comm_size(globComm, &nbp);
    // On interroge le communicateur global pour connaître l'identifiant qui
    // m'a été attribué ( en tant que processus ). Cet identifiant est compris
    // entre 0 et nbp-1 ( nbp étant le nombre de processus qui ont été lancés par
    // l'utilisateur )
    int rank;
    MPI_Comm_rank(globComm, &rank);
    // Création d'un fichier pour ma propre sortie en écriture :
    std::stringstream fileName;
    fileName << "Output" << std::setfill('0') << std::setw(5) << rank << ".txt";
    std::ofstream output(fileName.str().c_str());

    // Nombre total d'échantillons à lancer
    unsigned long totalSamples = 1000000000;
    // Nombre d'échantillons par processus
    unsigned long samplesPerProcess = totalSamples / nbp;

    // Mesure du temps d'exécution séquentiel
    double start_time_seq = omp_get_wtime();
    double pi_seq = approximate_pi_sequential(samplesPerProcess);
    double end_time_seq = omp_get_wtime();
    double time_seq = end_time_seq - start_time_seq;

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(15);
        std::cout << "Sequential Approximation of Pi = " << pi_seq << std::endl;
        std::cout << "Sequential Execution Time = " << time_seq << " seconds" << std::endl;
    }

    // Mesure du temps d'exécution parallèle avec différents nombres de threads
    for (int num_threads = 1; num_threads <= 8; num_threads *= 2) {
        omp_set_num_threads(num_threads);

        double start_time_par = omp_get_wtime();
        double pi_par = approximate_pi_parallel(samplesPerProcess);
        double end_time_par = omp_get_wtime();
        double time_par = end_time_par - start_time_par;

        double speedup = time_seq / time_par;

        if (rank == 0) {
            std::cout << "Parallel Approximation of Pi with " << num_threads << " threads = " << pi_par << std::endl;
            std::cout << "Parallel Execution Time with " << num_threads << " threads = " << time_par << " seconds" << std::endl;
            std::cout << "Speedup with " << num_threads << " threads = " << speedup << std::endl;
        }
    }

    output.close();
    // A la fin du programme, on doit synchroniser une dernière fois tous les processus
    // afin qu'aucun processus ne se termine pendant que d'autres processus continue à
    // tourner. Si on oublie cet instruction, on aura une plantage assuré des processus
    // qui ne seront pas encore terminés.
    MPI_Finalize();
    return EXIT_SUCCESS;
}