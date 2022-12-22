'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import os
import numpy as np
from pixel_classification.generate_rgb_data import read_pixels


class PixelClassifier():

    #Class Variables definitions

    theta_red           = 0
    theta_green         = 0
    theta_blue          = 0
    mu_red              = np.empty(3)
    mu_green            = np.empty(3)
    mu_blue             = np.empty(3)
    covariance_red      = np.empty((3,3))
    covariance_green    = np.empty((3,3))
    covariance_blue     = np.empty((3,3))


    def __init__(self):
        '''
        Initilize your classifier with any parameters and attributes you need
        '''

        #Load the variables from the text files and initialize them
        self.theta_red, self.theta_green, self.theta_blue    = np.loadtxt('pixel_classification/TrainingData_theta.txt')
        self.mu_red, self.mu_green, self.mu_blue             = np.loadtxt('pixel_classification/TrainingData_mu.txt')
        self.covariance_red                                  = np.loadtxt('pixel_classification/TrainingData_cov_red.txt')
        self.covariance_green                                = np.loadtxt('pixel_classification/TrainingData_cov_green.txt')
        self.covariance_blue                                 = np.loadtxt('pixel_classification/TrainingData_cov_blue.txt')

        pass

    def classify(self, X):

        '''
        Classify a set of pixels into red, green, or blue
        Inputs:
        X: n x 3 matrix of RGB values
        Outputs:
        y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively

        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE

        # Just a random classifier for now
        # Replace this with your own approach

        #Empty vector y to store the result of each pixel
        y = np.empty(len(X))

        for i in range(len(X)):

            #Calculation of Gaussian PDF for Red, Green and Blue Pixel basis the training model parameters calculated
            gaussian_PDF_Red = np.log(self.theta_red) - ( np.log(2*np.pi*np.linalg.det(self.covariance_red)) )/2 - ( np.dot(np.dot( ((X[i,:] - self.mu_red).T), np.linalg.inv(self.covariance_red) ) , (X[i,:] - self.mu_red)) )/2

            gaussian_PDF_Green = np.log(self.theta_green) - ( np.log(2*np.pi*np.linalg.det(self.covariance_green)) )/2 - ( np.dot(np.dot( ((X[i,:] - self.mu_green).T), np.linalg.inv(self.covariance_green) ) , (X[i,:] - self.mu_green)) )/2

            gaussian_PDF_Blue = np.log(self.theta_blue) - ( np.log(2*np.pi*np.linalg.det(self.covariance_blue)) )/2 - ( np.dot(np.dot( ((X[i,:] - self.mu_blue).T), np.linalg.inv(self.covariance_blue) ) , (X[i,:] - self.mu_blue)) )/2

            #Determination if the given pixel is Red, Green or Blue and accordingly store the value in vector y
            if(max(gaussian_PDF_Red, gaussian_PDF_Green, gaussian_PDF_Blue) == gaussian_PDF_Red):
                y[i] = 1

            elif(max(gaussian_PDF_Red, gaussian_PDF_Green, gaussian_PDF_Blue) == gaussian_PDF_Green):
                y[i] = 2

            elif(max(gaussian_PDF_Red, gaussian_PDF_Green, gaussian_PDF_Blue) == gaussian_PDF_Blue):
                y[i] = 3

        # YOUR CODE BEFORE THIS LINE

        ################################################################

        return y

    #Function to calculate model parameters from the training data
    def train_model(self):

        # Determination of RGB pixel values for each colour from the training set
        folder = 'data/training'
        X1 = read_pixels(folder+'/red')
        X2 = read_pixels(folder+'/green')
        X3 = read_pixels(folder+'/blue')

        #Calculation of theta (Probability) for each colour basis the samples given in the training set
        theta_Red     = len(X1)/(len(X1) + len(X2) + len(X3))
        theta_Green   = len(X2)/(len(X1) + len(X2) + len(X3))
        theta_Blue    = len(X3)/(len(X1) + len(X2) + len(X3))

        #Calculation of mu (Mean) of RGB pixels for each colour basis the samples given in the training set
        mu_Red          = (np.sum(X1, axis = 0)) / (len(X1))
        mu_Green        = (np.sum(X2, axis = 0)) / (len(X2))
        mu_Blue         = (np.sum(X3, axis = 0)) / (len(X3))

        #Calculation of Covariance matrix of RGB pixels for each colour basis the samples given in the training set
        covariance_Red      = (np.cov(X1.T).T)
        covariance_Green    = (np.cov(X2.T).T)
        covariance_Blue     = (np.cov(X3.T).T)

        #Save the model parameters in the text file
        np.savetxt('TrainingData_theta.txt', [theta_Red, theta_Blue, theta_Green], fmt = "%s")
        np.savetxt('TrainingData_mu.txt', [mu_Red, mu_Green, mu_Blue], fmt = "%s")
        np.savetxt('TrainingData_cov_red.txt', covariance_Red, fmt = "%s")
        np.savetxt('TrainingData_cov_green.txt', covariance_Green, fmt = "%s")
        np.savetxt('TrainingData_cov_blue.txt', covariance_Blue, fmt = "%s")

        pass
