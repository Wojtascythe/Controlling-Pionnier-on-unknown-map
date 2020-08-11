import json
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

dataFile1 = 'map_round.json' #10
dataFile2 = 'map_boxes_0.json' #4
dataFile3 = 'map_boxes_1.json' #9

json_data = open(dataFile1)
data = json.load(json_data)
x = np.arange(0, 512)
theta = (np.pi / 512) * (x - 256)
laserDistance = 0.12

# Wczytanie danych
skan = data[0]["scan"]
skanpose = data[0]["pose"]

mydata = []
poses = []
X = []
Y = []
polar = []
angle = []

for j in range(0, 10):
    skan = data[j]["scan"]
    currentpose = data[j]["pose"]
    for i in range(0, len(theta)):
        if np.isfinite(skan[i]):
            r = skan[i]
            xp = r * math.cos(theta[i]) + laserDistance
            yp = r * math.sin(theta[i])
            X.append(xp * math.cos(np.deg2rad(currentpose[2])) - yp * math.sin(np.deg2rad(currentpose[2])) + currentpose[0] + 6)
            Y.append(xp * math.sin(np.deg2rad(currentpose[2])) + yp * math.cos(np.deg2rad(currentpose[2])) + currentpose[1] + 6)
            polar.append(skan[i])
            angle.append(theta[i])

print(len(X))
print(len(Y))

plt.scatter(X, Y, 1)
plt.show()

class Map():
    def __init__(self, x_size, y_size, resolution=0.05):
        self.x_size = x_size
        self.y_size = y_size
        self.resolution = resolution
        self.occProb = 0.85
        self.freeProb = 1 - self.occProb
        self.log_occupied = np.log10(self.occProb / (1 - self.occProb))
        self.log_free = np.log(self.freeProb / (1 - self.freeProb))
        self.map = np.zeros([x_size, y_size])
        self.treshold = 2000
        self.mesh = []

    def visualize(self):
        plt.imshow(1 - 1/(1 + np.exp(self.map)), interpolation='nearest', cmap='Blues')
        plt.colorbar()          
        plt.show()

    def update_log_odds(self):
        for i in range(0, len(X)):
            self.mesh.append((int(X[i]/self.resolution), int(Y[i]/self.resolution)))
        #print(self.mesh)
        #self.mesh = set(self.mesh)
        i = 0
        prev = 0
        for pt in self.mesh:
            if 0 <= pt[0] < self.x_size and 0 <= pt[1] < self.y_size:
                if self.map[pt[0], pt[1]] < self.treshold:
                    self.map[pt[0], pt[1]] = self.map[pt[0], pt[1]] + self.log_occupied
        self.visualize()

if _name_ == "_main_":
    env_map = Map(250, 200)
    env_map.update_log_odds()
    #env_map.visualize()