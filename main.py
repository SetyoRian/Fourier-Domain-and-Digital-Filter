import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, exp

img = cv2.imread('Yamaha.jpg', 0)


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def ideal_lpf(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 1
    return base


def ideal_hpf(D0, imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 0
    return base


def butter_lpf(D0, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base


def butter_hpf(D0, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base


def gaussian_lpf(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base


def gaussian_hpf(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base


def spectrum():
    #   Fourier transform
    f = np.fft.fft2(img)
    #  Centered
    fshift = np.fft.fftshift(f)
    # logarithmic transformation
    magnitude_spectrum = np.log(1 + np.abs(fshift))
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # magnitude_spectrum = np.log(np.abs(fshift))
    phase = np.angle(fshift)
    f_ishift = np.fft.ifftshift(fshift)
    # inverse Fourier transform
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    plt.subplot(221), plt.imshow(img, cmap='gray'),
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray'),
    plt.title('Magnitude Spectrum'), plt.axis('off')
    plt.subplot(223), plt.imshow(phase, cmap='gray'),
    plt.title('Phase Spectrum'), plt.axis('off')
    plt.subplot(224), plt.imshow(img_back, cmap='gray'),
    plt.title('After Invers FFT'), plt.axis('off')
    plt.show()


def ideal():
    #   Fourier transform
    f = np.fft.fft2(img)
    #  Centered
    fshift = np.fft.fftshift(f)
    # Low Pass Call
    LowPass = ideal_lpf(50, img.shape)
    HighPass = ideal_hpf(50, img.shape)
    # Centered multiply Low Pass
    LowPassCenter = fshift * LowPass
    HighPassCenter = fshift * HighPass
    # Decentralize
    LowPass = np.fft.ifftshift(LowPassCenter)
    HighPass = np.fft.ifftshift(HighPassCenter)
    # Inverse Low Pass
    inverse_LowPass = np.fft.ifft2(LowPass)
    inverse_HighPass = np.fft.ifft2(HighPass)
    #
    inverse_LowPass = 1 + np.abs(inverse_LowPass)
    inverse_HighPass = 1 + np.abs(inverse_HighPass)
    plt.subplot(321), plt.imshow(img, cmap='gray'),
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(322), plt.imshow(np.log(1+np.abs(fshift)), cmap='gray'),
    plt.title('Magnitude Spectrum'), plt.axis('off')
    plt.subplot(323), plt.imshow(np.log(1+np.abs(LowPassCenter)), cmap='gray'),
    plt.title('Low Pass Filter'), plt.axis('off')
    plt.subplot(324), plt.imshow(np.log(1+np.abs(HighPassCenter)), cmap='gray'),
    plt.title('High Pass Filter'), plt.axis('off')
    plt.subplot(325), plt.imshow(inverse_LowPass, cmap='gray'),
    plt.title('Ideal LPF Image'), plt.axis('off')
    plt.subplot(326), plt.imshow(inverse_HighPass, cmap='gray'),
    plt.title('Ideal HPF Image'), plt.axis('off')
    plt.show()

def butterworth():
    #   Fourier transform
    f = np.fft.fft2(img)
    #  Centered
    fshift = np.fft.fftshift(f)
    # Low Pass Call
    LowPass = butter_lpf(50, img.shape, 10)
    HighPass = butter_hpf(50, img.shape, 10)
    # Centered multiply Low Pass
    LowPassCenter = fshift * LowPass
    HighPassCenter = fshift * HighPass
    # Decentralize
    LowPass = np.fft.ifftshift(LowPassCenter)
    HighPass = np.fft.ifftshift(HighPassCenter)
    # Inverse Low Pass
    inverse_LowPass = np.fft.ifft2(LowPass)
    inverse_HighPass = np.fft.ifft2(HighPass)
    #
    inverse_LowPass = 1 + np.abs(inverse_LowPass)
    inverse_HighPass = 1 + np.abs(inverse_HighPass)
    plt.subplot(321), plt.imshow(img, cmap='gray'),
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(322), plt.imshow(np.log(1+np.abs(fshift)), cmap='gray'),
    plt.title('Magnitude Spectrum'), plt.axis('off')
    plt.subplot(323), plt.imshow(np.log(1+np.abs(LowPassCenter)), cmap='gray'),
    plt.title('Low Pass Filter'), plt.axis('off')
    plt.subplot(324), plt.imshow(np.log(1+np.abs(HighPassCenter)), cmap='gray'),
    plt.title('High Pass Filter'), plt.axis('off')
    plt.subplot(325), plt.imshow(inverse_LowPass, cmap='gray'),
    plt.title('Butterworth LPF Image'), plt.axis('off')
    plt.subplot(326), plt.imshow(inverse_HighPass, cmap='gray'),
    plt.title('Butterworth HPF Image'), plt.axis('off')
    plt.show()


def gaussian():
    #   Fourier transform
    f = np.fft.fft2(img)
    #  Centered
    fshift = np.fft.fftshift(f)
    # Low Pass Call
    LowPass = gaussian_lpf(50, img.shape)
    HighPass = gaussian_hpf(50, img.shape)
    # Centered multiply Low Pass
    LowPassCenter = fshift * LowPass
    HighPassCenter = fshift * HighPass
    # Decentralize
    LowPass = np.fft.ifftshift(LowPassCenter)
    HighPass = np.fft.ifftshift(HighPassCenter)
    # Inverse Low Pass
    inverse_LowPass = np.fft.ifft2(LowPass)
    inverse_HighPass = np.fft.ifft2(HighPass)
    #
    inverse_LowPass = 1 + np.abs(inverse_LowPass)
    inverse_HighPass = 1 + np.abs(inverse_HighPass)
    plt.subplot(321), plt.imshow(img, cmap='gray'),
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(322), plt.imshow(np.log(1+np.abs(fshift)), cmap='gray'),
    plt.title('Magnitude Spectrum'), plt.axis('off')
    plt.subplot(323), plt.imshow(np.log(1+np.abs(LowPassCenter)), cmap='gray'),
    plt.title('Low Pass Filter'), plt.axis('off')
    plt.subplot(324), plt.imshow(np.log(1+np.abs(HighPassCenter)), cmap='gray'),
    plt.title('High Pass Filter'), plt.axis('off')
    plt.subplot(325), plt.imshow(inverse_LowPass, cmap='gray'),
    plt.title('Gaussian LPF Image'), plt.axis('off')
    plt.subplot(326), plt.imshow(inverse_HighPass, cmap='gray'),
    plt.title('Gaussian HPF Image'), plt.axis('off')
    plt.show()


while 1:
    print("Choose your option : ")
    print("1. Spectrum")
    print("2. Ideal Filter")
    print("3. Butterworth Filter")
    print("4. Gaussian Filter")
    print("5. Exit")
    print("-> ")
    ch = int(input())
    if ch == 1:
        spectrum()
    elif ch == 2:
        ideal()
    elif ch == 3:
        butterworth()
    elif ch == 4:
        gaussian()
    elif ch == 5:
        exit()
    else:
        print("Invalid Option !!!")
