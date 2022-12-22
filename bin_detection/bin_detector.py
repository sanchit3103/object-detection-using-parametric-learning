'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import os, cv2
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
from bin_detection.roipoly.roipoly import RoiPoly

class BinDetector():

	#Class Variables definitions

	theta_Desired_Blue 			= 0
	theta_Undesired_Blue 		= 0
	theta_Other_Colors			= 0
	mu_Desired_Blue				= np.empty(3)
	mu_Undesired_Blue			= np.empty(3)
	mu_Other_Colors				= np.empty(3)
	covariance_Desired_Blue		= np.empty((3,3))
	covariance_Undesired_Blue	= np.empty((3,3))
	covariance_Other_Colors		= np.empty((3,3))


	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''

		#Load the variables from the text files and initialize them
		self.theta_Desired_Blue, self.theta_Undesired_Blue, self.theta_Other_Colors 	= np.loadtxt('bin_detection/ModelParameters_theta.txt')
		self.mu_Desired_Blue, self.mu_Undesired_Blue, self.mu_Other_Colors				= np.loadtxt('bin_detection/ModelParameters_mu.txt')
		self.covariance_Desired_Blue													= np.loadtxt('bin_detection/ModelParameters_cov_desired_blue.txt')
		self.covariance_Undesired_Blue													= np.loadtxt('bin_detection/ModelParameters_cov_undesired_blue.txt')
		self.covariance_Other_Colors													= np.loadtxt('bin_detection/ModelParameters_cov_other_colors.txt')
		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture,
			call other functions in this class if needed

			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Convert the image to RGB Format
		img = img.astype(np.float64)/255 #Normalization of pixel values in range of 0 - 1
		mask_bin = np.empty([img.shape[0], img.shape[1]]) #Empty matrix to store the mask of the bin

		for i in range(img.shape[0]):
			for j in range(img.shape[1]):

				gaussian_PDF_Desired_Blue 	= np.log(self.theta_Desired_Blue) - ( np.log(2*np.pi*np.linalg.det(self.covariance_Desired_Blue)) )/2 - ( np.dot(np.dot( ((img[i, j] - self.mu_Desired_Blue).T), np.linalg.inv(self.covariance_Desired_Blue) ) , (img[i, j] - self.mu_Desired_Blue)) )/2

				gaussian_PDF_Undesired_Blue = np.log(self.theta_Undesired_Blue) - ( np.log(2*np.pi*np.linalg.det(self.covariance_Undesired_Blue)) )/2 - ( np.dot(np.dot( ((img[i, j] - self.mu_Undesired_Blue).T), np.linalg.inv(self.covariance_Undesired_Blue) ) , (img[i, j] - self.mu_Undesired_Blue)) )/2

				gaussian_PDF_Other_Colors 	= np.log(self.theta_Other_Colors) - ( np.log(2*np.pi*np.linalg.det(self.covariance_Other_Colors)) )/2 - ( np.dot(np.dot( ((img[i, j] - self.mu_Other_Colors).T), np.linalg.inv(self.covariance_Other_Colors) ) , (img[i, j] - self.mu_Other_Colors)) )/2

				if(max(gaussian_PDF_Desired_Blue, gaussian_PDF_Undesired_Blue, gaussian_PDF_Other_Colors) == gaussian_PDF_Desired_Blue):
					mask_bin[i, j] = 255

				else:
					mask_bin[i, j] = 0

		mask_bin = mask_bin.astype(np.uint8)
		mask_img = mask_bin

		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed

			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE

		#Obtain the contours present in the mask image of the bin
		contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		boxes = [] #Initialize empty boxes list

		#Loop to filter the unwanted contours and draw a bounding box around the bin
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)
			aspect_ratio = float(w/h)
			if h > w and aspect_ratio >= 0.5:
				#cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
				boxes.append([x,y,x+w,y+h])

		# YOUR CODE BEFORE THIS LINE
		################################################################

		return boxes

	def generate_rgb_data(self, folder):

		#Declaration of Variables
		X 		= np.empty([70000,3])
		i 		= 0

		#Loop to collect RGB values of ROI
		for filename in os.listdir(folder):
			img 		= cv2.imread(os.path.join(folder,filename))
			img 		= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			#Display Image
			fig, ax 	= plt.subplots()
			ax.imshow(img)
			#Use RoiPoly for labelling
			my_roi 		= RoiPoly(fig=fig, ax=ax, color='r')
			#Get Image mask
			mask 		= my_roi.get_mask(img)

			#Display labelled region and image mask
			fig, (ax1, ax2) = plt.subplots(1, 2)
			fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])
			ax1.imshow(img)
			ax2.imshow(mask)

			plt.show(block=True)

			#Save the RGB Values of the ROI in X
			X[i:i+img[mask,:].shape[0]]		= img[mask==1].astype(np.float64)/255
			#Set the index of the matrix X according to the shape of the mask
			i 								= i + img[mask,:].shape[0]

		#Truncate the RGB matrix to remove all zeros
		X = X[~np.all(X == 0, axis=1)]
		#print(np.shape(X))

		return X

	def rgb_Data_Training_Samples(self):

		folder = 'data/training/'
		#Collection of RGB values for desired-blue samples and store in a text file
		rgb_desired_blue 	= self.generate_rgb_data(folder+'/Desired_Blue')
		np.savetxt('RGB_Data_Desired_Blue.txt', rgb_desired_blue, fmt = "%s")

		#Collection of RGB values for undesired-blue samples and store in a text file
		rgb_undesired_blue 	= self.generate_rgb_data(folder+'/Undesired_Blue')
		np.savetxt('RGB_Data_Undesired_Blue.txt', rgb_undesired_blue, fmt = "%s")

		#Collection of RGB values for samples of other colors and store in a text file
		rgb_other_colors 	= self.generate_rgb_data(folder+'/Other_Colors')
		np.savetxt('RGB_Data_Other_Colors.txt', rgb_other_colors, fmt = "%s")

		pass

	def calculate_Model_Parameters(self):

		X1 = np.loadtxt('RGB_Data_Desired_Blue.txt') #Load RGB data of Desired Blue from training samples
		X2 = np.loadtxt('RGB_Data_Undesired_Blue.txt') #Load RGB data of Undesired Blue from training samples
		X3 = np.loadtxt('RGB_Data_Other_Colors.txt') #Load RGB data of Other colours from training samples

		#Calculation of theta (Probability) for the RGB data collected for the 3 sample sets
		theta_desired_blue	 = len(X1)/(len(X1) + len(X2) + len(X3))
		theta_undesired_blue = len(X2)/(len(X1) + len(X2) + len(X3))
		theta_other_colors	 = len(X3)/(len(X1) + len(X2) + len(X3))

		#Calculation of mu (Mean) of RGB pixels for the 3 sample sets
		mu_desired_blue		 = (np.sum(X1, axis = 0)) / (len(X1))
		mu_undesired_blue	 = (np.sum(X2, axis = 0)) / (len(X2))
		mu_other_colors		 = (np.sum(X3, axis = 0)) / (len(X3))

		#Calculation of Covariance matrix of RGB pixels for the 3 sample sets
		covariance_desired_blue		 = (np.cov(X1.T).T)
		covariance_undesired_blue	 = (np.cov(X2.T).T)
		covariance_other_colors		 = (np.cov(X3.T).T)

		#Save the model parameters in the text file
		np.savetxt('ModelParameters_theta.txt', [theta_desired_blue, theta_undesired_blue, theta_other_colors], fmt = "%s")
		np.savetxt('ModelParameters_mu.txt', [mu_desired_blue, mu_undesired_blue, mu_other_colors], fmt = "%s")
		np.savetxt('ModelParameters_cov_desired_blue.txt', covariance_desired_blue, fmt = "%s")
		np.savetxt('ModelParameters_cov_undesired_blue.txt', covariance_undesired_blue, fmt = "%s")
		np.savetxt('ModelParameters_cov_other_colors.txt', covariance_other_colors, fmt = "%s")

		pass
