import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df1 = pd.read_csv(r'pc1.csv',header=None)
df2 = pd.read_csv(r'pc2.csv', header=None)
#Taking PC1 columnwise
X1=df1[0]
Y1=df1[1]
Z1=df1[2]
#Taking PC2 columnwise
X2=df2[0]
Y2=df2[1]
Z2=df2[2]
#Ransac Curve fitting involves taking n samples of the data at a time and fitting curve to it, and finding the maximum number of points the curve coincides.
#The best curve has most number of the points in the plane.
#This Process is done iteratively
#definiing Inline arrays for PC1 and PC2
inlinea=[]
inlineb=[]
coarray1=[]
coarray2=[]
#Threshhold is given as Zero-mean Gaussian noise with std. dev. σ: t**2=3.84*σ**2
thresh=np.sqrt(3.84 * np.var(df1))
thresh1=np.sqrt(3.84 * np.var(df2))
#Number of samples = log(1− p)/ log(1− (1− e)^s )
#P(success) = 0.99, P(outlier) = 0.5
N= int(np.log(1 - 0.99)/np.log(1 - (1 - 0.5)**4))
for i in range(N):
    #We choose random points in the dataset using np.random
    in1 = np.random.choice(range(len(df1[0])), size=3, replace=False)
    #using least squares for the selected random points
    x1, y1, z1 = df1[0][in1[0]], df1[1][in1[0]], df1[2][in1[0]]
    x2, y2, z2 = df1[0][in1[1]], df1[1][in1[1]], df1[2][in1[1]]
    x3, y3, z3 = df1[0][in1[2]], df1[1][in1[2]], df1[2][in1[2]]
    #Same as total least squares, the plane is defined as ax+by+cz=d
    # but here a b c are derieved using : a=[y2-y1] * [z2-z1] - [z2-z1][y3-y2]
    a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y2)
    b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z2)
    c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x2)
    d = -(a * x1 + b * y1 + c * z1)
    coarray1.append([a, b, c, d])
    #Doing similar operation for second data points
    in2 = np.random.choice(range(len(df1[0])), size=3, replace=False)
    x11, y11, z11 = df2[0][in2[0]], df2[1][in2[0]], df2[2][in2[0]]
    x22, y22, z22 = df2[0][in2[1]], df2[1][in2[1]], df2[2][in2[1]]
    x33, y33, z33 = df2[0][in2[2]], df2[1][in2[2]], df2[2][in2[2]]
    a1 = (y22 - y11) * (z33 - z11) - (z22 - z11) * (y33 - y22)
    b1= (z22 - z11) * (x33 - x11) - (x22 - x11) * (z33 - z22)
    c1= (x22- x11) * (y33 - y11) - (y22 - y11) * (x33 - x22)
    d1= -(a1 * x11 + b1 * y11 + c1 * z11)
    coarray2.append([a1, b1, c1, d1])
    #define inlier count as zero
    inline1 = 0
    inline2=0
    #For the first data point
    for j in range(len(df1[0])):
        x, y, z = df1[0][j], df1[1][j], df1[2][j]  
        dis1= (a * x + b * y + c * z + d)/(np.sqrt(a**2 + b**2 + c**2))
        if dis1 < thresh.all():
            inline1+= 1
    
    #For the second data point
    for m in range(len(df2[0])):
        xh, yh, zh = df2[0][j], df2[1][j], df2[2][j]
        dis2= (a1 * xh + b1 * yh + c1 * zh + d1)/(np.sqrt(a1**2 + b1**2 + c1**2))
        if dis2 < thresh1.all():
            inline2+=1
    inlinea.append(inline1)
    inlineb.append(inline2)
#take maximum number of points coinciding with the curve    
ind1= np.argmax(inlinea)
ind2= np.argmax(inlineb)
print("inlier points:",coarray1[ind1])
print("inlier points in second data:",coarray2[ind2])

#Plotting
model1 = coarray1[ind1]
model2=coarray2[ind2]
fig = plt.figure()
#Equation of new plane z=(-(ax+by)+d/c)
model11= (-(model1[0]*(X1) + model1[1]*(Y1)) + model1[3])/model1[2]
ax = plt.axes(projection='3d')
plt.title('RANSAC PC1.CSV')
ax.scatter3D(X1,Y1,Z1,c=Z1)
ax.plot_trisurf(X1,Y1,model11)
#For second Model
model12 = (-(model2[0]*(X2) + model2[1]*(Y2)) + model2[3])/model2[2]
fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('RANSAC PC2.CSV')
ax.scatter3D(X2,Y2,Z2,c=Z2)
ax.plot_trisurf(X2,Y2,model12)   
plt.show()