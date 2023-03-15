import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df1 = pd.read_csv(r'pc1.csv',header=None)
df2 = pd.read_csv(r'pc2.csv', header=None)
# print(df1[0])
#Calculating Covariacne 
x=np.array(df1)
y=np.array(df2)
n=x.shape
# print(n)
# x0=x.shape[0]
# x1=x.shape[1]
#To calculate Covariance, we can directly use the fomrula 
#S=1/n* sum ((x-xmean)*(x-xmean).T)
covariance=np.zeros((n[1],n[1]))
avg=[]
#initialize average
for i in range(n[1]):
    for values in x:
        mean=sum([values[i]])/n[0]
    avg.append(mean)
print("Average is: ",avg)
#Once we get the mean, we need to find the covariance as the summation of X value and its mean multiplied by
#transpose of difference X and its mean. 
#looping it columnwise
for i in range(n[1]):
    for j in range(n[1]):
            covs=0
            for m in x:
                covs+=np.dot((m[i]-avg[i]),(m[j]-avg[j]).T)
            covs=covs/(n[0]-1)
            covariance[i,j]=covs
print("The Covariance Matrix:\n",covariance)
print("*********************************************Answer for the 2(a) 2rd Part*******************************\n")
#Magnitude and direction is just the vectorized form of the covariance matrix as both are vector quantities.
#Calculating eigen vector and eigen values can give the magnitude and direction
lambd,M= np.linalg.eig(covariance)
print("Eigen Values are [Magnitude]:",lambd)
print("Eigen Vectors are [Direction]:\n",M)
print("*********************************************Answer for the 2(b) 1rd Part*******************************\n")
#Implementation of OLS [Ordinary least Squares]
#Theoratically OLS is given by the formula B= (X.T X)^-1 . (X.T Y)
#For our simplified Understanding
#Taking PC1 columnwise
X1=df1[0]
Y1=df1[1]
Z1=df1[2]
#Taking PC2 columnwise
X2=df2[0]
Y2=df2[1]
Z2=df2[2]
# #Initially Plotting the figure
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(X1,Y1,Z1,c=Z1);
# fig2 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(X2,Y2,Z2)
#taking z=f(x,y) and z=ax+by+c, we construct a least square algorithm
xvec=np.vstack([X1, Y1, np.ones(len(X1))]).T
# print(xvec)
x1vec=np.vstack([X2, Y2, np.ones(len(X2))]).T
#This constructs a matrix with X1, Y1 and 1's as values to find the equation of the curve
#Finding inverse of this matrix according to the formula
inverse=np.linalg.pinv(np.dot(xvec.T,xvec))
inverse2=np.linalg.pinv(np.dot(x1vec.T,x1vec))
#Substituting in the Formula
lsq=np.dot(inverse,np.dot(xvec.T,Z1))
lsq1=np.dot(inverse,np.dot(x1vec.T,Z2))
print("**************LEAST SQUARES******** ")
print("Least Squares for the first data points: ",lsq)
print("Least Squares for the second data points: ",lsq1)
#Now Curve fitting
#We know the original equation of the plane is ax+by+c=z as z=f(x,y)
#Source :https://www.analyzemath.com/line/equation-of-line.html
# equn=ax+by+c
mpl=X1*lsq[0]+Y1*lsq[1]+lsq[2]
mpl2=X2*lsq1[0]+Y2*lsq1[1]+lsq1[2]
fig1 = plt.figure()
ax1 = plt.axes(projection='3d')
plt.title("Curve for PC 1")
ax1.scatter3D(X1,Y1,Z1,c=X1)
ax1.plot_trisurf(X1,Y1,mpl)
fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
plt.title("Curve for PC 2")
ax2.scatter3D(X2,Y2,Z2,c=X2)
ax2.plot_trisurf(X2,Y2,mpl2,color='red')
plt.show()
print("\n")
print("*******************************TOTAL LEAST SQUARES************************** ")
#To Calculate Total  Least Squares, we can use the covariance matrix we obtained 
#M is the eigen vector for first data points, let's calculate Covariance matrix for second
n1=y.shape
covariance2=np.zeros((n1[1],n1[1]))
avg1=[]
#initialize average
for i in range(n1[1]):
    for value in y:
        mean1=sum([value[i]])/n1[0]
    avg1.append(mean1)
# print("Average is: ",avg1)
#Once we get the mean, we need to find the covariance as the summation of X value and its mean multiplied by
#transpose of difference X and its mean. 
#looping it columnwise
for i in range(n1[1]):
    for j in range(n1[1]):
            covs1=0
            for val in x:
                covs1+=np.dot((val[i]-avg1[i]),(val[j]-avg1[j]).T)
            covs1=covs1/(n1[0]-1)
            covariance2[i,j]=covs1
# print("The Covariance Matrix:\n",covariance)
lambd1,M2= np.linalg.eig(covariance2)
# print("Eigen Values are [Magnitude]:",lambd1)
# print("Eigen Vectors are [Direction]:\n",M2)
#Cartesian Equation of the line is given as ax+by+cz=d, where a b c are co-efficients of Eigen Vector. 
#a=M[0][0],b=M[1][0]and C=M[2][0]. We can choose any column of the eigen vector as solution.
#In TLS, we find X-Xmean, now to find xmean, we use
d=M[0][0]*np.mean(X1)+M[1][0]*np.mean(Y1)+M[2][0]*np.mean(Z1)
#Similarly for the second data points
d1=M2[0][0]*np.mean(X2)+M2[1][0]*np.mean(Y2)+M2[2][0]*np.mean(Z2)
#The equation of the new plane fitting the curve is given as Z= (-(ax+by)+d)/c
z=(1/M[2][0])*(-(X1*M[0][0]+Y1*M[1][0])+d)
z1=(1/M2[2][0])*(-(X2*M2[0][0]+Y2*M2[1][0])+d1)
# print("Equation of Plane1 : ",z)
# print("Equation of Plane2 : ",z1)
#Plotting
fig11 = plt.figure()
ax11 = plt.axes(projection='3d')
plt.title("Total Least Square Curve for PCL 1")
ax11.scatter3D(X1,Y1,Z1,c=Z1)
ax11.plot_trisurf(X1,Y1,z)
# plt.plot(X1,Y1,z)
fig12= plt.figure()
ax12= plt.axes(projection='3d')
plt.title("Total Least Square Curve for PCL 2")
ax12.scatter3D(X2,Y2,Z2,c=Z2)
ax12.plot_trisurf(X2,Y2,z1,color='g')
plt.show()