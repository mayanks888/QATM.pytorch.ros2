import cv2


path='/home/mayank_sati/Desktop/git/2/AI/QATM/data/sample/gwm_975.jpg'
# path='cool.jpg'

img = cv2.imread(path)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgray = cv2.merge((imgray,imgray,imgray))
cv2.imwrite('cool.jpg',img2)
cv2.imwrite('cool1.jpg',imgray)