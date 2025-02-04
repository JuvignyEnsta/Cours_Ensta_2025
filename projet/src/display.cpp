#include <cassert>
#include <stdexcept>
#include <string> 
#include "display.hpp"

using namespace std::string_literals;

std::shared_ptr<Displayer> Displayer::unique_instance{nullptr};

Displayer::Displayer( std::uint32_t t_width, std::uint32_t t_height )
{
    // Initialisation du contexte pour SDL :
    // ----------------------------------------------------------------------------------------------------------------
    // Puisqu'on a qu'une seule instance pour la fenêtre,
    // on peut se permettre d'initialiser le contexte de
    // SDL dans la création de la fenêtre à afficher !
    if ( SDL_Init( SDL_INIT_EVERYTHING) < 0 )
    {
        std::string err_msg = "Erreur lors de l'initialisation de SDL : "s + std::string(SDL_GetError());
        throw std::runtime_error(err_msg);
    }
    // Création de la fenêtre à afficher sur l'ecran (ainsi que le renderer associé):
    // ----------------------------------------------------------------------------------------------------------------
    SDL_CreateWindowAndRenderer(t_width, t_height, 0, &m_pt_window, &m_pt_renderer);
    if (m_pt_window == nullptr) // On vérifie si la fenêtre a bien été créée
    {
        std::string err_msg = "Erreur lors de la création de la fenêtre : "s + std::string(SDL_GetError() );
        throw std::runtime_error(err_msg);
    }
    if (m_pt_renderer == nullptr) // On vérifie si la fenêtre a bien été créée
    {
        std::string err_msg = "Erreur lors de la création du moteur de rendu : "s + std::string(SDL_GetError() );
        throw std::runtime_error(err_msg);
    }
    // Récupération de la surface sur laquelle on va dessiner et afficher dans la fenêtre 
    // ----------------------------------------------------------------------------------------------------------------
    m_pt_surface = SDL_GetWindowSurface( m_pt_window );
    if (m_pt_surface == nullptr) // On vérifie qu'on a bien récupéré la surface associée à la fenêtre
    {
        std::string err_msg = "Erreur lors de la récupération de la surface : "s + std::string(SDL_GetError() );
        throw std::runtime_error(err_msg);
    }
}
// ====================================================================================================================
Displayer::~Displayer()
{
    // Destruction du moteur de rendu :
    SDL_DestroyRenderer(m_pt_renderer);
    // Destruction de la fenêtre ( et de la surface implicitement puisque associée à la fenêtre )
    SDL_DestroyWindow( m_pt_window );
    // On détruit également le contexte SDL
    SDL_Quit();
}
// ====================================================================================================================
void
Displayer::update( std::vector<std::uint8_t> const & vegetation_global_map,
                   std::vector<std::uint8_t> const & fire_global_map )
{
    int w, h;
    SDL_GetWindowSize(m_pt_window, &w, &h );
    SDL_SetRenderDrawColor(m_pt_renderer, 0,0,0, 255);
    SDL_RenderClear(m_pt_renderer);
    for (int i = 0; i < h; ++i )
      for (int j =  0; j < w; ++j )
      {
        SDL_SetRenderDrawColor(m_pt_renderer, fire_global_map[j + w*i], vegetation_global_map[j + w*i], 0, 255);
        SDL_RenderDrawPoint(m_pt_renderer, j, h-i-1); 
      }
    SDL_RenderPresent(m_pt_renderer);
}
// ####################################################################################################################
//                      Définition des méthodes statiques associées au pattern singleton utilisé
std::shared_ptr<Displayer> 
Displayer::init_instance( std::uint32_t t_width, std::uint32_t t_height )
{
    assert( ( "L'initialisation de l'instance ne doit etre appele qu'une seule fois !" && (unique_instance == nullptr) ) );
    unique_instance = std::make_shared<Displayer>(t_width, t_height);
    return unique_instance;
}
// ====================================================================================================================
std::shared_ptr<Displayer> 
Displayer::instance()
{
    assert( ( "Il faut initialiser l'instance avant tout !" && (unique_instance != nullptr) ) );
    return unique_instance;
}
