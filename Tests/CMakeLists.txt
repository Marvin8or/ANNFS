project(Tests)
file(GLOB Tests_src "*.hpp" "*.cpp")
add_library(Tests ${Tests_src})
