import numpy as np
from scipy import misc

name, grid = "matrix1", [15, 14, 3]
#name, grid = "matrix2", [33, 38, 3]

arr = misc.imread(name + ".png")
arr = arr.mean(2)

Dx = (arr.shape[0]+10)/grid[0]
Dy = (arr.shape[1]+10)/grid[1]

print (name + ".png"), "->", (name + "-out.png"), (name + "-out.csv")

new_arr = np.zeros(grid)
for x in xrange(grid[0]):
    for y in xrange(grid[1]):
        Ox, Oy = int(0.35*Dx), int(0.35*Dy)
        new_arr[x,y,:] = arr[Dx*x + Ox, Dy*y + Oy]

out = misc.imresize(new_arr, [5*Dx,5*Dy,1], interp="nearest")
misc.imsave(name + "-out.png", out)

amin = arr.min()
amax = arr.max()

with file(name + "-out.csv", "w") as f:
    for y in xrange(grid[1]):
        for x in xrange(grid[0]):
            if x > 0:
                f.write(", ")
            f.write("%.2f" % (1 - (new_arr[x,y,0] - amin) / (amax-amin)))
        f.write("\n")
