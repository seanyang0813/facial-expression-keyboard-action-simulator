import math
origin_x=0
origin_y=0
angle=3.14/6
curx=0
cury=10
x=origin_x+math.cos(angle)*(curx-origin_x)-math.sin(angle)*(cury- origin_y)
y=origin_y+math.sin(angle)*(curx-origin_x)+math.cos(angle)*(cury- origin_y)
print(x)
print(y)
