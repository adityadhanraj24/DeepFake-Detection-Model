import cv2 as cv

img=cv.imread("Photos/Cat_large.jpg")

cv.imshow("Cat",img)
cv.waitKey(0)