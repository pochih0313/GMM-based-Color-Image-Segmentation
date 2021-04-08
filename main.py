# %%
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM
from matplotlib import pyplot as plt

n = 1 # figure index

def prediction(model, test_data, shape):
    global n
    plt.figure(n)
    n += 1

    labels = model.predict(test_data)
    result = labels.reshape(shape[0], shape[1])
    plt.imshow(result, plt.cm.gray)
    plt.show()

if __name__ == '__main__':
    ### Read data
    img_soccer1 = cv2.imread("soccer1.jpg")
    mask_soccer1 = cv2.imread("soccer1_mask.png", 0)
    img_soccer2 = cv2.imread("soccer2.jpg")
# %% Scenario1
    print("Scenario1:")


    training_image = cv2.bitwise_and(img_soccer1, img_soccer1, mask=mask_soccer1)
    # plt.imshow(img_soccer1)
    # plt.show()
    training_image = training_image.reshape((-1, 3))
    training_data = []

    for pixel in training_image:
        if (pixel[0] == 0 & pixel[1] == 0 & pixel[2] == 0):
            continue
        else:
            training_data.append(pixel)
    training_data = np.array(training_data)
    # print(training_data)

    model1 = GMM(n_components=2, covariance_type='full').fit(training_data)
    
    test_image = img_soccer1.reshape((-1,3))
    shape = img_soccer1.shape

    prediction(model1, test_image, shape)
# %% Scenario2
    print("Scenario2:")
    
    test_image = img_soccer2.reshape((-1,3))
    shape = img_soccer2.shape
    prediction(model1, test_image, shape)
# %% Scenario3
    print("Scenario3:")

# %%
