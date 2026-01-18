Dataset is really small, so even small naive architectures are able to represent it quite well.
However, as it is that small it also overfits a lot when training from scratch, so multiple
regularizations must be selected. Some options are:
    - Data agumentation (added 100%)
    - Dropout (added 100% when head has more than one layer)
    - Weight decay (need to test in sweep later, last time was not really important)
    - Normalizations (for now, batch normalization, but later on sweep to check which one works best)

It seems that the main thing to do will be to tune the various hyperparameters, such as the
learning rate in order to get the best performance possible.

Adding data augmentation makes the small lenet train really slowly due to the added complexity of
the data. However, it seems that it stops increasing the train accuracy arround 0.5. This indicates
the need for a more complex arquitecture in order to reach higher accuracies.

Also, it seems that data augmentation alone seems to be good enough to regularize small models
from scratch, so no further regularization will be introduce aside from normalizations right now.
(Normalizations help train the networks faster)