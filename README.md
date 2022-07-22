# XOR-problem-ML
Following <a href="https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6Y7MdSCaIfsxc561QI0U0Tb" target="_blank">THIS</a> youtube course I implemented my own neural network and used it to solve the XOR problem.
The neural network consists of 
1 input layer containing 2 nodes
1 hidden layer containing 4 nodes
1 output layer containing 1 node


## Algorithms used
#### FeedForward algorithm.
#### Backpropagation algorithm
#### Sigmoid activation function

## Libraries used 
Numpy for matrix operations
Pygame for visual represation of the neural network output


### Output
![Screenshot (222)](https://user-images.githubusercontent.com/87566788/180405629-0258e8b8-7936-4230-9dea-1d820ded9862.png)


The previous graph consists of 64 * 64 squares each of 10 px size, the color of the square is computed using the guess provided by the neural network based on the square row and column following the equation : color = 255 * NN.Guess([x,y]). Thus, the closer the output is to 0 the darker the color.
