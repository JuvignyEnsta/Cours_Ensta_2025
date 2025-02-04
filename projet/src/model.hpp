#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include <unordered_map>

/**
 * @brief 
 * 
 */
class Model
{
public:
    struct LexicoIndices
    {
        unsigned row, column;
    };

    Model( double t_length, unsigned t_discretization, std::array<double,2> t_wind,
           LexicoIndices t_start_fire_position, double t_max_wind = 60. );
    Model( Model const & ) = delete;
    Model( Model      && ) = delete;
    ~Model() = default;

    Model& operator = ( Model const & ) = delete;
    Model& operator = ( Model      && ) = delete;

    bool update();

    unsigned geometry() const { return m_geometry; }
    std::vector<std::uint8_t> vegetal_map() const { return m_vegetation_map; }
    std::vector<std::uint8_t> fire_map() const { return m_fire_map; }
    std::size_t time_step() const { return m_time_step; }

private:
    std::size_t   get_index_from_lexicographic_indices( LexicoIndices t_lexico_indices  ) const;
    LexicoIndices get_lexicographic_from_index        ( std::size_t t_global_index ) const;

    double m_length;                    // Taille du carré représentant le terrain (en km)
    double m_distance;                  // Taille d'une case du terrain modélisé
    std::size_t m_time_step;            // Dernier numéro du pas de temps calculé
    unsigned m_geometry;                // Taille en nombre de cases de la carte 2D
    std::array<double,2> m_wind{0.,0.}; // Vitesse et direction du vent suivant les axes x et y en km/h
    double m_wind_speed;                // Norme euclidienne de la vitesse du vent
    double m_max_wind; //+ Vitesse à partir de laquelle le feu ne peut pas se propager dans le sens opposé à celui du vent.
    std::vector<std::uint8_t> m_vegetation_map, m_fire_map;
    double p1{0.}, p2{0.};
    double alphaEastWest, alphaWestEast, alphaSouthNorth, alphaNorthSouth;

    std::unordered_map<std::size_t, std::uint8_t> m_fire_front;
};
