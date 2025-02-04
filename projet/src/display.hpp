#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#if defined(__linux__)
#  include <SDL2/SDL.h>
#endif
class Displayer
{
public:
    Displayer( std::uint32_t width, std::uint32_t height );
    Displayer( Displayer const & ) = delete;
    Displayer( Displayer      && ) = delete;
    ~Displayer();

    Displayer& operator = ( Displayer const & ) = delete;
    Displayer& operator = ( Displayer      && ) = delete;

    void update( std::vector<std::uint8_t> const & vegetation_global_map,
                 std::vector<std::uint8_t> const & fire_global_map );

    static std::shared_ptr<Displayer> init_instance( std::uint32_t t_width, std::uint32_t t_height );
    static std::shared_ptr<Displayer> instance();

private:
    static std::shared_ptr<Displayer> unique_instance;


    SDL_Renderer *m_pt_renderer{nullptr};
    SDL_Surface  *m_pt_surface{nullptr};
    SDL_Window   *m_pt_window {nullptr};
};
