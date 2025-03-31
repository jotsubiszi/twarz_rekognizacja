import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import tkinter.filedialog as fd


def Laplacian(img, blur_size=5):
    kernel = np.ones((blur_size, blur_size), np.float32) / (blur_size ** 2)
    blurred = cv2.filter2D(img, -1, kernel)
    return cv2.Laplacian(blurred, cv2.CV_64F)


def Sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return sobelx + sobely


def convEdges(img):
    img_ = cv2.GaussianBlur(img, ksize=(11, 11), sigmaX=0)
    vertKernel = np.vstack(([1, 1, 1], [0, 0, 0], [1, 1, 1]))
    horiKernel = np.vstack(([1, 0, 1], [1, 0, 1], [1, 0, 1]))
    vertConv = sc.ndimage.convolve(img_, vertKernel, mode="nearest")
    horiConv = sc.ndimage.convolve(img_, horiKernel, mode="nearest")
    out = np.abs(vertConv - horiConv)

    openKernel = np.ones((4, 4), np.float32)
    print(openKernel)
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, openKernel)


def main():
    file = fd.askopenfilename()
    if file == ():
        print("Error: invalid file")
        exit(1)
    image = cv2.imread(file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_shifted = np.roll(image_gray, 2)

    filter_kernel = np.ones((5, 5), np.float32) / 25

    edges = cv2.normalize(image_gray - image_shifted, None,
                          alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    edges_blurred = np.abs(cv2.filter2D(edges, -1, filter_kernel))
    edges_lap = np.abs(Laplacian(image_gray))
    edges_conv = np.abs(convEdges(image_gray))
    edges_sobel = np.abs(Sobel(image_gray))
    cv2.destroyAllWindows()
    plt.subplot(221)
    plt.imshow(edges_blurred, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.title("Edge subtraction + blur (5x5)")

    plt.subplot(222)
    plt.imshow(edges_lap, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.title("Laplacian (filtering 5x5)")

    plt.subplot(223)
    plt.imshow(edges_conv, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.title("Convolution")

    plt.subplot(224)
    plt.imshow(edges_sobel, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.title("Sobel (hori + vert sum)")

    plt.show()


if __name__ == "__main__":
    main()
