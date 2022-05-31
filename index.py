import swapper as sp
import cv2

face = cv2.imread("image-1.jpg")
head = cv2.imread("image-2.jpg")

changed = sp.change(face, head)

cv2.imwrite("result.jpg", changed)