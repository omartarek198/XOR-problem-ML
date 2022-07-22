import ANN

import numpy as np
NN = ANN.NeuralNetwork(2,4,1,0.1)
import sys

import pygame
#creating data
inputs = []
inputs.append([0,0])
inputs.append([0,1])
inputs.append([1,1])
inputs.append([1,0])
labels = []
labels.append(0)
labels.append(1)
labels.append(0)
labels.append(1)

#train


for n in range (10000):

    for i in range(len(labels)):
        NN.train(inputs[i], labels[i])








# Configuration

pygame.init()
fps = 60
fpsClock = pygame.time.Clock()
width, height = 640, 640
screen = pygame.display.set_mode((width, height))








def DrawScene():
    while True:
        screen.fill((20, 20, 20))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            for i in range (64):
                for k in range (64):

                    v1 = i/64
                    v2 = k /64

                    R = 255
                    G = 255
                    B = 255
                    pred = NN.Guess([v1,v2])
                    print ("input:" , [v1,v2])

                    print ("pred:" , pred)
                    R = R * pred
                    R = int(R)
                    B = B * pred
                    B = int(B)
                    G = G * pred
                    G = int(G)
                    print (G)

                    # Drawing Rectangle
                    pygame.draw.rect(screen, [R,G,B], pygame.Rect( 10 * k, 10 * i , 10, 10))



            pygame.display.flip()
            fpsClock.tick(fps)



DrawScene()
