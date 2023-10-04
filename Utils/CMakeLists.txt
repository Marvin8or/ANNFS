project(Utils)
file(GLOB Utils_src "*.h" "*.cpp")

add_library(Utils ${Utils_src})
