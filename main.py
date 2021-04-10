# %%
import numpy as np
import cv2
import joblib
from sklearn.mixture import GaussianMixture as GMM
from matplotlib import pyplot as plt

n = 1 # figure index

def prediction(model, test_data, shape):
    global n
    plt.figure(n)
    n += 1

    labels = model.predict(test_data)
    result = labels.reshape(shape[0], shape[1])

    print('segmentation result')
    plt.imshow(result)
    plt.show()

    return labels

def find_pixel_accuracy(labels, mask, shape, n):
    ground_truth = []
    result = []

    for pixel in mask:
        if pixel == 255:
            ground_truth.append(1)
        else:
            ground_truth.append(0)
    
    field_index = np.argmax(np.bincount(labels))
    for label in labels:
        if label == field_index:
            result.append(1)
        else:
            result.append(0)
    
    print("pixel detection")
    show_result = np.array(result)
    show_result = show_result.reshape(shape[0], shape[1])
    plt.imshow(show_result, plt.cm.gray)
    plt.savefig(n)
    plt.show()

    count = 0
    for i in range(len(result)):
        if result[i] == ground_truth[i]:
            count+=1
    pixel_accuracy = count / len(result)
    print('pixel accuracy: ' + str(pixel_accuracy))

# %% Read Files
if __name__ == '__main__':
    img_soccer1 = cv2.imread("soccer1.jpg")
    img_soccer2 = cv2.imread("soccer2.jpg")

    mask_soccer1 = cv2.imread("soccer1_mask.png", 0)
    mask_soccer2 = cv2.imread("soccer2_mask.png", 0)

    model1 = joblib.load('model1')
    model2 = joblib.load('model2')
# %% Scenario1
    print("Scenario1:")

    test_image = img_soccer1.reshape((-1,3))
    shape = img_soccer1.shape
    
    mask1 = mask_soccer1.reshape(-1)
    labels1 = prediction(model1, test_image, shape)
    find_pixel_accuracy(labels1, mask1, shape, str(1))
# %% Scenario2
    print("Scenario2:")
    
    test_image = img_soccer2.reshape((-1,3))
    shape = img_soccer2.shape

    mask2 = mask_soccer2.reshape(-1)
    labels2 = prediction(model1, test_image, shape)
    find_pixel_accuracy(labels2, mask2, shape, str(2))
# %% Scenario3-1
    print("Scenario3-soccer1:")

    test_image = img_soccer1.reshape((-1,3))
    shape = img_soccer1.shape
    
    labels3_1 = prediction(model2, test_image, shape)
    find_pixel_accuracy(labels3_1, mask1, shape, '3-1')
# %% Scenario3-2
    print("Scenario3-soccer2:")
    
    test_image = img_soccer2.reshape((-1,3))
    shape = img_soccer2.shape
    
    labels3_2 = prediction(model2, test_image, shape)
    find_pixel_accuracy(labels3_2, mask2, shape, '3-2')

# %%
# def prediction(model_1, model_0, test_data, shape):
#     global n
#     plt.figure(n)
#     n += 1

#     field_prob = model_1.score_samples(test_data)
#     notfield_prob = model_0.score_samples(test_data)

#     result = []
#     for i in range(len(test_data)):
#         if (field_prob[i] > notfield_prob[i]):
#             result.append(1)
#         else:
#             result.append(0)

#     result = np.array(result)
#     result = result.reshape(shape[0], shape[1])

#     plt.imshow(result, plt.cm.gray)
#     plt.show()

# %% Scenario1
    # playfield_data = []
    # notplayfield_data = []
    
    # for i in range(len(mask_soccer1)):
    #     if (mask_soccer1[i] == 0):
    #         notplayfield_data.append(training_image[i])
    #     else:
    #         playfield_data.append(training_image[i])
    
    # playfield_data = np.array(playfield_data)
    # notplayfield_data = np.array(notplayfield_data)

    # #prediction(model1_1, model1_0, test_image, shape)