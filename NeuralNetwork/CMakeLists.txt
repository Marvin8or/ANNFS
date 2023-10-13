project(NeuralNetwork)
file(GLOB NeuralNetwork_src "*.h" "*.cpp")
add_library(NeuralNetwork ${NeuralNetwork_src})
