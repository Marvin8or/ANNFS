project(Math)
file(GLOB LinearAlgebra_src "LinearAlgebra/*.h" "LinearAlgebra/*.cpp")

add_library(Math ${LinearAlgebra_src})
