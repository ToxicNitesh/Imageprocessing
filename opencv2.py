import cv2
import numpy as np
from matplotlib import pyplot as plt
i=3

while(i):
    i=int(input("1 for bluring image \n 2 for sharpening\n 0 for stop"))
    if i==2:
        image=cv2.imread("D:\cat.jpg",cv2.IMREAD_GRAYSCALE)
        kernel=np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image_sharp=cv2.filter2D(image,-1,kernel)
        
        plt.imshow(image_sharp,cmap="gray"),plt.axis("off")
        plt.show()
        
    elif i==1:
        img = cv2.imread('D:\cat.jpg')

        blur = cv2.blur(img,(5,5))

        plt.subplot(121),plt.imshow(img),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
        plt.xticks([]), plt.yticks([])
        plt.show()
    elif i==3:
        img_src = cv2.imread('D:\cat.jpg')


        kernel = np.array([[0.0, -1.0, 0.0], 
                           [-1.0, 4.0, -1.0],
                           [0.0, -1.0, 0.0]])

        kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

        #filter the source image
        img_rst = cv2.filter2D(img_src,-1,kernel)

        #save result image
        cv2.imwrite('result.jpg',img_rst)


