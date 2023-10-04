project(Math)
file(GLOB LinearAlgebra_src "LinearAlgebra/Matrix.h")

add_library(Math ${LinearAlgebra_src})
