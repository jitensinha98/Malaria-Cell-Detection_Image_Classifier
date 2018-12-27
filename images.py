import os
import matplotlib.pyplot as plt
import cv2

path_infected = 'Train/Parasitized'
path_normal = 'Train/Uninfected'

for img in os.listdir(path_infected) :
	image=os.path.join(path_infected,img)
	im=cv2.imread(image)
	plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
	plt.title('Parasitized')
	plt.show()
	break;

for imgr in os.listdir(path_normal) :
	imager=os.path.join(path_normal,imgr)
	imr=cv2.imread(imager)
	plt.imshow(cv2.cvtColor(imr, cv2.COLOR_BGR2RGB))
	plt.title('Uninfected')
	plt.show()
	break;
