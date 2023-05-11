project(LinearAlgebra)
file(GLOB LinearAlgebra_src "*.h" "*.cpp")
file(GLOB Operations_src "operations/*.h" "operations/*.cpp")

add_library(LinearAlgebra ${LinearAlgebra_src} ${Operations_src})
