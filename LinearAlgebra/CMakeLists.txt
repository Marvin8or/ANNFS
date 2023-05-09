project(LinearAlgebra)
file(GLOB LinearAlgebra_src "*.h" "*.cpp")
add_library(LinearAlgebra ${LinearAlgebra_src})