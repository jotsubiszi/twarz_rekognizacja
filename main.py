import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter.filedialog as fd
from scipy import signal

WINDOW_NAME = "test"

scharr=np.array([[-3-3j, 0-10j, +3-3j],
                [-10+0j, 0+0j, +10+0j],
                [-3+3j, 0+10j, +3+3j]])

kernel=np.array([[0.1,0.1,0.1],
                 [0.1, 0.5,0.1],
                 [0.1,0.1,0.1]])


def main():
    file = fd.askopenfilename()
    if file == ():
        print("Error: invalid file")
        exit(1)

    image = cv2.imread(file)
    image_filter = cv2.filter2D(image,-1, kernel)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_filter = cv2.filter2D(image_gray,-1, kernel)
    grad=signal.convolve2d(image_filter, scharr, boundary='symm', mode='same')

    fig, (ax_orig, ax_mag) = plt.subplots(1,2)
    ax_orig.imshow(image)
    ax_orig.set_title("obraz oryginalny")
    ax_orig.set_axis_off()

    ax_mag.imshow(np.absolute(grad), cmap='grey')
    ax_mag.set_title("gradient magnitude")
    ax_mag.set_axis_off()

    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, image)
    while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(0)

    plt.show()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
