# %%
import joblib
import itertools
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM
from matplotlib import pyplot as plt

def plot_bic(training_image, n):
    min_bic = np.infty
    bic = []
    n_components_range = range(1, 10)
    cvtypes = ['spherical', 'tied', 'diag', 'full']
    for cvtype in cvtypes:
        for n_components in n_components_range:
            gmm_model = GMM(n_components=n_components, covariance_type=cvtype).fit(training_image)

            bic.append(gmm_model.bic(training_image))
            # if bic[-1] < min_bic:
            #     min_bic = bic[-1]
            #     model = gmm_model

    # Plot the BIC scores
    bic = np.array(bic)
    colors = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
    bars = []

    plt.figure(figsize=(12, 6))
    for i, (cvtype, color) in enumerate(zip(cvtypes, colors)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                    (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    plt.legend([b[0] for b in bars], cvtypes)
    fig_name = 'BIC_' + str(n) + '.jpg'
    plt.savefig(fig_name)
    plt.show()
    
# %% Model1: Read & Plot BIC
if __name__ == '__main__':
    img_soccer1 = cv2.imread("soccer1.jpg")
    img_soccer2 = cv2.imread("soccer2.jpg")
    mask_soccer1 = cv2.imread("soccer1_mask.png", 0)
    mask_soccer2 = cv2.imread("soccer2_mask.png", 0)

# %% Model1: Plot BIC to select the model
    training_image = img_soccer1.reshape((-1, 3))
    plot_bic(training_image, 1)
# %% Model1: Training (The best choice: 2 components with covariance type='full')
    model1 = GMM(n_components=2, covariance_type='full').fit(training_image)
    joblib.dump(model1, 'model1')
# %% Model2: Plot BIC to select the model
    training_image2 = img_soccer2.reshape((-1, 3))
    training_image = np.concatenate([training_image, training_image2])
    plot_bic(training_image, 2)
# %% Model2: Training (The best choice: 2 components with covariance type='full')
    model2 = GMM(n_components=2, covariance_type='full').fit(training_image)
    joblib.dump(model2, 'model2')
# %%
