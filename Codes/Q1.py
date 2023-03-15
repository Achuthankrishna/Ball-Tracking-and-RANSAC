import cv2
import numpy as np
import matplotlib.pyplot as plt

a = cv2.VideoCapture('ball.mov')
x = []
y = []

while a.isOpened():
    ret, frame = a.read()
    if not ret:
        break
    red = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(red, (5, 5), 0)
    lred = np.array([0, 180, 100])
    ured = np.array([9, 255, 255])
    mask = cv2.inRange(red, lred, ured)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    rc = cv2.bitwise_and(frame, frame, mask=mask)
    #When the pixel values per frame of the mask is nonzero and not 'Nan'
    if np.count_nonzero(mask) > 0:
        nonzero = np.nonzero(mask)
        centroid = np.mean(nonzero, axis=1)
        x.append(centroid[1])
        y.append(centroid[0])

    cv2.imshow("frame", frame)
    cv2.imshow("Gaga", rc)
    # print(nonzero)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

a.release()
cv2.destroyAllWindows()

plt.imshow(mask, cmap='gray')
plt.plot(x, y, 'o', markersize=5)
plt.show()
# ******************************************* 1.2**********************************************
#Curve fitting
# #assume parabola equation as ax**2+bx+c
xm=np.array(x)
xvec=np.vstack([xm**2,xm, np.ones(len(x))]).T
inverse=np.linalg.pinv(np.dot(xvec.T,xvec))
lsq=np.dot(inverse,np.dot(xvec.T,y))
mpl=xm**2*lsq[0]+xm*lsq[1]+lsq[2]
plt.scatter(x, y)
plt.plot(x,mpl,c='r')
plt.show()
#equation of the curve
print("*********************************************Answer for the 2rd Part*******************************\n")
print("The Curve equation is :",lsq[0],"x**2","+",lsq[1],"x","+",lsq[2])
print("\n")
# ******************************************* 1.3**********************************************
#Given that first landing point of Y co-ordinate is shifted to 300 Px
#Yl=Y[0]+300
yy=np.array(y)
yland = y[0] + 300
#from curve equation we know :ax**2+bx+c, so roots = -b +- (sqrt(b*b-4ac))/2a
root = np.roots([ lsq[0], lsq[1], lsq[2] -yland])
xland = np.real(root[0])
print("*********************************************Answer for the 3rd Part*******************************\n")
print("The landing X co-ordinate roots are:" ,root)
print("The landing X co-ordinate:" ,xland)