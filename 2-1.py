import cv2
im = cv2.imread(filename)
cv2.namedWindow("cat")
cv2.imshow("cat",im)
cv2.waitKey()
cv2.destroyWindow()
print(in)
cv2.imwrite("cat",filename)