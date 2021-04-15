import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

goldenRatio = (1 + 5**0.5) / 2

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
        
        self.X = []
        self.Y = []
        self.Z = []
        
        # равномерное распределение точек на боковой поверхности параболоида
        I = np.arange(0, dots, dtype=float) + random.uniform(0, 1)
        U = np.arccos(2 * I**2 / dots**2)
        U /= math.pi/2
        U *= float(math.sqrt(self.h))
        V = 2 * math.pi * goldenRatio * I
        
        self.X = U*np.cos(V)
        self.Y = U*np.sin(V)
        self.Z = (U*U)
        # удаление невалидных элементов
        self.Z = self.Z[~np.isnan(self.Z)]
        self.Y = self.Y[~np.isnan(self.Y)]
        self.X = self.X[~np.isnan(self.X)]
    
    # функция расчёта расстояние до точки на параболоиде
    def checkDistance(self, checkPoint):
        for i in range(0, len(self.z)):
            savedPoint = ParaboloidPoint(0,0)
            savedPoint.setXYZ(self.x[i], self.y[i], self.z[i])
            if checkPoint.distance(savedPoint) <= self.neighborhood:
                return True
        return False

# Основание параболоида
class ParaboloidCap:
    def __init__(self, height, dots):
        self.h = height
        self.r = math.sqrt(height)
        self.X = []
        self.Y = []
        self.Z = []

        # генерация равномерного распределения точек на основании параболоида
        I = np.arange(0, dots, dtype=float) + random.uniform(0, 1)
        U = 2 * math.pi * goldenRatio * I
        R = np.sqrt(I) / math.sqrt(dots) * self.r
        self.X = R * np.cos(U)
        self.Y = R * np.sin(U)
        self.Z = np.repeat(self.h, dots)

# Параболоид с основанием
class FullParaboloid:
    # объединение основания и боковой поверхности
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


# Функция генерации сферы (для отображение на графике)
def drawSphere(xCenter, yCenter, zCenter, r):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)

    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

def getAreaRatio(height):
    r = math.sqrt(height)
    S_cap = math.pi*r*r
    u = 1 + 4*r*r
    S_paraboloidSide = (math.pi / 6) * (u**(3/2) - 1)
    return (S_cap / (S_cap + S_paraboloidSide))

# Функция генерации случайной точки
def getRandomParaboloidPoint(height):
    side = random.uniform(0, 1)
    # расчёт отношения площади основания к площади поверхности параболоида
    areaRatio = getAreaRatio(height)
    r = math.sqrt(height)

    if side < areaRatio:
      randomU = random.uniform(0, 1) * 2 * math.pi
      x = r * random.uniform(0, 1) * math.cos(randomU)
      y = r * random.uniform(0, 1) * math.sin(randomU)
      randomCapPoint = ParaboloidPoint(0,0)
      randomCapPoint.setXYZ(x,y,height)
      return randomCapPoint
    else:
      randomU = random.uniform(0, 1) * math.sqrt(height)
      randomV = random.uniform(0, 1) * 2 * math.pi
      randomParaboloidPoint = ParaboloidPoint(randomU, randomV)
      return randomParaboloidPoint


# Параметры игры
s = 200
e = 0.01
h = 1
iterNum = 10

# Инициализация фигур
p1 = Paraboloid(s, e, h)
c1 = ParaboloidCap(h, int(getAreaRatio(h)*s))
fp1 = FullParaboloid(p1.X, p1.Y, p1.Z, c1.X, c1.Y, c1.Z, e)

# Инициализация графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Генерация случайных точек второго игрока и 
# расчёт расстояния до точек первого игрока
random_points = []
wins = 0
for i in range(0, iterNum):
    if i % 1000 == 0:
        print("Done:", i, "Wins:", wins)
    # Генерация случайной точки
    randParaboloidPoint = getRandomParaboloidPoint(h)
    random_points.append(randParaboloidPoint)
    
    # если расстояние до одной из точек, меньше, чем e, то 1 игроку присуждается победа
    if fp1.checkDistance(randParaboloidPoint):
        wins += 1
print("Game Value:", (wins/iterNum))

# Генерация и отображение параболоида
u = np.linspace(0, math.sqrt(h), 50)
v = np.linspace(0, 2*np.pi, 50)
U, V = np.meshgrid(u, v)
Z = U**2
X, Y = U * np.cos(V), U * np.sin(V)
ax.plot_surface(X, Y, Z, alpha=0.3)
#cmap=plt.cm.YlGnBu_r

# Отображение точек первого игрока на боковой поверхности параболоида
for (xi, yi, zi, ri) in zip(p1.X, p1.Y, p1.Z, np.full(len(p1.X), e)):
    (xs, ys, zs) = drawSphere(xi, yi, zi, ri)
    ax.plot_wireframe(xs, ys, zs, color="r")

# Отображение точек первого игрока на основании
for (xi, yi, zi, ri) in zip(c1.X, c1.Y, c1.Z, np.full(len(c1.X), e)):
    (xs, ys, zs) = drawSphere(xi, yi, zi, ri)
    #ax.plot_wireframe(xs, ys, zs, color="y")

# Отображение точек второго игрока на фигуре
for i in random_points:
    (xs, ys, zs) = drawSphere(i.x, i.y, i.z, 0.05)
    #ax.plot_wireframe(xs, ys, zs, color="g")

plt.show()
