import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from scipy import signal
import os
from cleandoc.settings import MEDIA_ROOT, STATICFILES_DIRS
import shutil


def remove_rotation(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(image)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
    print("Rotated")
    return rotated


def remove_noise(image):
    test_files = image
    model = tf.keras.models.load_model(STATICFILES_DIRS[0] + "/model_32.h5")
    # model.summary()
    img = get_chunks(test_files)
    pred_chunks = model.predict(img.reshape(-1, 32, 32, 1))
    pred_chunks = pred_chunks.reshape(img.shape)
    # show_chunks(pred_chunks)
    the_page = reassemble_chunks(pred_chunks)
    # cv2.imwrite("test.png", np.squeeze(the_page))
    # print(the_page)
    # cv2.imwrite("img.jpg", np.squeeze(np.squeeze(results[0])))
    # plt.figure(figsize=(20,20))
    plt.imsave(image, the_page, cmap="gray")
    # plt.show()


def get_chunks(file):
    page = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # getting the hight and width of the image, old_page_height and old_page_width
    oph, opw = page.shape[:2]
    # getting new height and width
    # to fit the chunks in the image use max - (max%32) to get rid of the remaining.
    # it is a fast solution we can use for now
    nph, npw = oph-(oph % 32), opw-(opw % 32)
    row_chunks = nph//32  # numober of rows
    col_chunks = npw//32  # number of chunks
    rc = 0  # row counter
    cc = 0  # column counter
    # the structure is convertible between chunks and the initial image
    img_chunks = np.ones((row_chunks, col_chunks, 32, 32, 1), dtype="float32")
    # the paper shredder
    for row in range(0, nph, 32):
        cc = 0
        for col in range(0, npw, 32):
            nimg = page[row:row+32, col:col+32]/255.
            nimg = np.array(nimg).reshape(32, 32, 1)
            try:
                img_chunks[rc, cc] = nimg
            except:
                print(rc, cc)
            cc += 1
        rc += 1
    return img_chunks


def show_chunks(chunks):
    for row in chunks:
        plt.figure(figsize=(10, 10))
        for i, chunk in enumerate(row):
            plt.subplot(1, len(row), i+1)
            plt.imshow(chunk.reshape(32, 32), "gray")
            plt.axis("OFF")
        plt.show()


def reassemble_chunks(chunks):
    # getting the page size
    oph, opw = chunks.shape[0]*32, chunks.shape[1]*32
    the_page = np.ones((oph, opw), dtype="float32")
    for r, row in enumerate(chunks):
        r = r*32
        for c, chunk in enumerate(row):
            c = c*32
            the_page[r:r+32, c:c+32] = chunk.reshape(32, 32)
    return the_page


def denoise_image(inp):
    bg = signal.medfilt2d(inp, 11)
    mask = inp < bg - 0.1
    return np.where(mask, inp, 1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    args = vars(ap.parse_args())

    img = (np.asarray(cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE))/255.0)
    image = denoise_image(img)
    # imag = np.asarray(image*255.0, dtype=np.uint8)
    cv2.imwrite("output.png", np.asarray(image*255.0, dtype=np.uint8))

    # image = cv2.imread("output.png", cv2.IMREAD_GRAYSCALE)
    remove_noise("output.png")
    image = cv2.imread("output.png", cv2.IMREAD_GRAYSCALE)
    image = remove_rotation(image)
    cv2.imwrite("output.png", image)


def doc(image):
    out_path = MEDIA_ROOT+"/processed.jpg"
    image = MEDIA_ROOT+image
    shutil.copy(image, STATICFILES_DIRS[0])
    image = STATICFILES_DIRS[0]+"/uploaded.jpg"
    img = (np.asarray(cv2.imread(image, cv2.IMREAD_GRAYSCALE))/255.)
    image = denoise_image(img)
    # imag = np.asarray(image*255.0, dtype=np.uint8)
    cv2.imwrite(out_path, np.asarray(image*255.0, dtype=np.uint8))

    # image = cv2.imread("output.png", cv2.IMREAD_GRAYSCALE)
    remove_noise(out_path)
    image = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
    image = remove_rotation(image)
    cv2.imwrite(out_path, image)
    return True
