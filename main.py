import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter.filedialog as fd

WINDOW_NAME = "test"


def main():
    file = fd.askopenfilename()
    if file == ():
        print("Error: invalid file")
        exit(1)
    image = cv2.imread(file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, image_gray)
    while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
