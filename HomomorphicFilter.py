import cv2
import numpy as np
import math

"""
A way of looking at images is

image = illuminance * reflectance

Where illuminance is created by scene lighting conditions and reflectance is a property of the objects themselves.
We're interested in the reflectance. We first take the logarithm and get

log(image) = log(illuminance) + log(reflectance)

You can imagine that illuminance varies slowly over the image, i.e. is low frequency. Applying a high pass filter
does a good job of getting rid of it.

All these functions do is take the log of the intensity values, applies a high-pass filter, and then brings us back
with exponentiation
"""

def createGaussianFilter(shape, sigma):
    sigSq = sigma**2
    const = 1/(2 * math.pi * sigSq)
    lowPass = np.zeros(shape, dtype=np.float32)
    centerX = math.ceil(shape[1]/2)
    centerY = math.ceil(shape[0]/2)
    for i in range(shape[1]):
        for j in range(shape[0]):
            distSq = (centerX - i)**2 + (centerY - j)**2
            lowPass[j][i] = const * math.exp(-distSq/(2*sigSq))

    high = np.amax(lowPass)
    low = np.amin(lowPass)
    lowPass -= low
    lowPass *= (1.0 / (high - low))
    highPass = 1 - lowPass
    return highPass

def getReflectance(img):
    img = np.log1p(np.array(img, dtype="float") / 255)

    h,w = img.shape
    optHeight = cv2.getOptimalDFTSize(h)
    optWidth = cv2.getOptimalDFTSize(w)

    padded = cv2.copyMakeBorder(img, 0, optHeight - h, 0, optWidth - w, cv2.BORDER_CONSTANT, value = 0)
    dft = cv2.dft(np.float32(padded),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)

    highPass = createGaussianFilter(dft.shape, 20)
    highPassed = cv2.mulSpectrums(dft, highPass, flags=0)
    highPassed = np.fft.fftshift(highPassed)

    highFiltered = cv2.dft(highPassed, flags=cv2.DFT_INVERSE)

    min, max = np.amin(highFiltered), np.amax(highFiltered)
    highFiltered = (highFiltered - min) / (max - min)

    recovered = np.expm1(highFiltered)

    final = recovered[:h,:w,0]
    low, high, _, _ = cv2.minMaxLoc(final)
    final = final * (1.0/(high-low)) + ((-low)/(high-low))
    final = np.uint8(final * 255)

    return final