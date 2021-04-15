import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

# Точка параболоида
class ParaboloidPoint:
    def __init__(self, u, v):
        self.x = u * math.cos(v)
        self.y = u * math.sin(v)
        self.z = u * u
    def setXYZ(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z 
    def distance(self, p):
        return math.sqrt((self.x - p.x) * (self.x - p.x) + 
                         (self.y - p.y) * (self.y - p.y) + 
                         (self.z - p.z) * (self.z - p.z))

# Параболоид
class Paraboloid:
    def __init__(self, dots, neighborhood, h):
        
        self.dots = dots
        self.neighborhood = neighborhood
        self.h = h
        
        self.x = []
        self.y = []
        self.z = []
        
        # равномерное распределение точек на боковой поверхности параболоида
        indices = np.arange(0, dots, dtype=float) + random.uniform(0, 1)
        u = np.arccos(2 * indices**2 / dots**2)
        v = math.pi * (1 + 5**0.5) * indices
        u /= math.pi/2
        u *= float(math.sqrt(self.h))
        self.z, self.x, self.y = (u*u), (u*np.cos(v)), u*np.sin(v)
        
        # удаление невалидных элементов
        self.z = self.z[~np.isnan(self.z)]
        self.y = self.y[~np.isnan(self.y)]
        self.x = self.x[~np.isnan(self.x)]
        

        self.points = []
        for i in range(0, dots):
            u = random.uniform(0, 1)
            v = random.uniform(0, 1) * 2 * math.pi
            newPoint = ParaboloidPoint(u, v)
            self.points.append(newPoint)
    
    # функция расчёта расстояние до точки
    def checkDistance(self, checkPoint):
        for i in range(0, len(self.z)):
            savedPoint = ParaboloidPoint(0,0)
            savedPoint.setXYZ(self.x[i], self.y[i], self.z[i])
            if checkPoint.distance(savedPoint) <= self.neighborhood:
                return True
        return False

# Крышечка параболоида
class ParaboloidCap:
    def __init__(self, height, dots):
        self.h = height
        self.r = math.sqrt(height)
        self.x = []
        self.y = []
        self.z = []

        # генерация равномерного распределения точек на основании параболоида
        golden_angle = math.pi * (3 - math.sqrt(5))

        for i in range(0, dots):
            self.z.append(self.h)
            theta = i * golden_angle
            r = math.sqrt(i) / math.sqrt(dots) * self.r
            self.x.append(r * math.cos(theta))
            self.y.append(r * math.sin(theta))

# Параболоид с крышечкой
class FullParaboloid:
  def __init__(self, X_par, Y_par, Z_par, X_cap, Y_cap, Z_cap, e):
    self.e = e
    self.X = X_par
    self.Y = Y_par
    self.Z = Z_par
    for (i, j, k) in zip(X_cap, Y_cap, Z_cap):
      self.X = np.append(self.X, i)
      self.Y = np.append(self.Y, j)
      self.Z = np.append(self.Z, k)
  def checkDistance(self, checkPoint):
        for i in range(0, len(self.Z)):
            savedPoint = ParaboloidPoint(0,0)
            savedPoint.setXYZ(self.X[i], self.Y[i], self.Z[i])
            if checkPoint.distance(savedPoint) <= self.e:
                return True
        return False

# Функция генерации случайной точки
def getRandomParaboloidPoint(height):
    side = random.uniform(0, 1)
    r = math.sqrt(height)
    if side < 0.3:
      theta = random.uniform(0, 1) * 2 * math.pi
      x = r * random.uniform(0, 1) * math.cos(theta)
      y = r * random.uniform(0, 1) * math.sin(theta)
      randomCapPoint = ParaboloidPoint(0,0)
      randomCapPoint.setXYZ(x,y,height)
      return randomCapPoint
    else:
      randomU = random.uniform(0, 1) * math.sqrt(height)
      randomV = random.uniform(0, 1) * 2 * math.pi
      randomParaboloidPoint = ParaboloidPoint(randomU, randomV)
      return randomParaboloidPoint

# Функция генерации сферы
def drawSphere(xCenter, yCenter, zCenter, r):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)

    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

# Параметры игры
dots = 6
neighborhood = 0.8
h = 1
iterates = 50

p1 = Paraboloid(dots, neighborhood, h)
c1 = ParaboloidCap(h, int(dots / 3))
fp1 = FullParaboloid(p1.x, p1.y, p1.z, c1.x, c1.y, c1.z, neighborhood)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Генерация случайных точек второго игрока и 
# расчёт расстояния до точек первого игрока
random_points = []
wins = 0
for i in range(0, iterates):
    if i % 1000 == 0:
        print("Done:", i, "Wins:", wins)
    randParaboloidPoint = getRandomParaboloidPoint(h)
    random_points.append(randParaboloidPoint)
    if fp1.checkDistance(randParaboloidPoint):
        wins += 1
print("Game Value:", (wins/iterates))

# Генерация и отображение параболоида
theta = np.linspace(0, math.sqrt(h), 50)
phi = np.linspace(0, 2*np.pi, 50)
Theta, Phi = np.meshgrid(theta, phi)
Z = Theta**2
X, Y = Theta * np.cos(Phi), Theta * np.sin(Phi)
ax.plot_surface(X, Y, Z, alpha=0.3)
#cmap=plt.cm.YlGnBu_r

# Отображение точек первого игрока на боковой поверхности параболоида
for (xi, yi, zi, ri) in zip(p1.x, p1.y, p1.z, np.full(len(p1.x), neighborhood)):
    (xs, ys, zs) = drawSphere(xi, yi, zi, ri)
    #ax.plot_wireframe(xs, ys, zs, color="r")

# Отображение точек первого игрока на основании
for (xi, yi, zi, ri) in zip(c1.x, c1.y, c1.z, np.full(len(c1.x), neighborhood)):
    (xs, ys, zs) = drawSphere(xi, yi, zi, ri)
    #ax.plot_wireframe(xs, ys, zs, color="y")

# Отображение точек второго игрока на фигуре
for i in random_points:
    (xs, ys, zs) = drawSphere(i.x, i.y, i.z, 0.05)
    ax.plot_wireframe(xs, ys, zs, color="g")

plt.show()
