# %%
import numpy
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

if __name__ == '__main__':
    ### Read data
    img_soccer1 = cv2.imread("soccer1.jpg")
    img_soccer2 = cv2.imread("soccer2.jpg")
# %%
    print("Scenario1:")

    training_image = img_soccer1.reshape((-1, 3))
    model1 = GMM(n_components=2, covariance_type='full').fit(training_image)
    
    test_image = training_image
    shape = img_soccer1.shape

    prediction(model1, test_image, shape)

    print("Scenario2:")
    
    test_image = img_soccer2.reshape((-1,3))
    shape = img_soccer2.shape
    prediction(model1, test_image, shape)
# %% Scenario3
    print("Scenario3:")
