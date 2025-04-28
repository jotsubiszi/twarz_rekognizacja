import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import tkinter.filedialog as fd


def Laplacian(img, blur_size=5):
    kernel = np.ones((blur_size, blur_size), np.float32) / (blur_size ** 2)
    blurred = cv2.filter2D(img, -1, kernel)  # filtr uśredniający
    return cv2.Laplacian(blurred, cv2.CV_64F)


def Sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return np.sqrt(sobelx ** 2 + sobely ** 2)


def Prewitt(img):
    img_ = np.astype(img, np.float32)
    horiKernel = np.vstack(
        ([1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]), dtype=np.float32)
    vertKernel = np.vstack(
        ([1., 0., -1.], [1., 0., -1.], [1., 0., -1.]), dtype=np.float32)
    vertConv = sc.ndimage.convolve(img_, vertKernel, mode="nearest")
    horiConv = sc.ndimage.convolve(img_, horiKernel, mode="nearest")
    out = np.sqrt(vertConv ** 2 + horiConv ** 2)
    return out


def normalise(a):
    return (a - np.min(a)) / np.ptp(a)


def main():
    file = fd.askopenfilename()
    if file == ():
        print("Error: invalid file")
        exit(1)
    image = cv2.imread(file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges_lap = Laplacian(image_gray)
    edges_prewitt = Prewitt(image_gray)
    edges_sobel = Sobel(image_gray)
    cv2.destroyAllWindows()
    plt.subplot(231)
    plt.imshow(image_gray, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.title("Original image")

    plt.subplot(232)
    plt.imshow(edges_lap, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.title("averaging filter 5x5 + Laplacian")

    plt.subplot(234)
    plt.imshow(edges_prewitt, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.title("Prewitt (Euclidian distance of hori and vert)")

    plt.subplot(235)
    plt.imshow(edges_sobel, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.title("Sobel (Euclidian distance of hori and vert)")

    sobel_norm = normalise(edges_sobel)
    prewitt_norm = normalise(edges_prewitt)
    plt.subplot(236)
    plt.imshow(np.abs(sobel_norm - prewitt_norm), cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.title("Difference between Sobel and Prewitt")

    plt.show()


def Haar():
    file = fd.askopenfilename(title="Select image")
    if file == ():
        print("Error: invalid file")
        exit(1)
    image = cv2.imread(file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_eqhist = cv2.equalizeHist(image_gray)  # wyrównywanie histogramu
    _image = np.copy(image_gray)
    _image_eqh = np.copy(image_eqhist)

    face_cascade = cv2.CascadeClassifier()

    if not face_cascade.load(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")):
        print("Error while loading cascade file")
        exit(2)

    faces = face_cascade.detectMultiScale(image_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        cv2.ellipse(image_gray, center, (w//2, h//2),
                    0, 0, 360, (255, 0, 255), 4)

    faces = face_cascade.detectMultiScale(image_eqhist)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        cv2.ellipse(image_eqhist, center, (w//2, h//2),
                    0, 0, 360, (255, 0, 255), 4)
    plt.subplot(221)
    plt.imshow(image_gray, cmap="gray")
    plt.title("Haar cascades")

    plt.subplot(222)
    plt.hist(_image.ravel(), 256)
    plt.title("Histogram")

    plt.subplot(223)
    plt.imshow(image_eqhist, cmap="gray")
    plt.title("Haar cascades (after histogram equalization)")

    plt.subplot(224)
    plt.hist(_image_eqh.ravel(), 256)
    plt.title("Histogram")
    plt.show()


if __name__ == "__main__":
    # main()
    Haar()
