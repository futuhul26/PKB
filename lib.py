import numpy as np

print_statistics(images, t_images, labels, t_labels):
    #number of training dataset
    len(labels)
    #number of test dataset
    len(t_labels)
    #number of class
    len(np.unique(labels))
    #number of instances per class on training
    for i in np.unique(labels):
        np.sum(labels==i)
    #number of instances per class on test dataset
    for i in np.unique(t_labels):
        np.sum(t_labels==i)
        
def hypothesis(images, labels):
    ekspo = []
    for i in range(np.unique(labels)):
        sumo = theta[0]
        for j in range(len(images.iloc[0])):
            sumo += np.sum(theta[i+1]*images.iloc[:,i])
        ekspo.append(np.exp(sumo))
    sumo = np.sum(ekspo)
    return ekspo/sumo        