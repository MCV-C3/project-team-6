IF SOMEONE THAT IS NOT ME READ THIS, READ IT AS IF IT WAS A DIARY, I WRITE WHILE DOING THE
EXPERIMENTS, SO IT'S WHAT I'M THINKING TO DO AT EACH STEP, NOT ANY RESULT OR SOMETHING LIKE THAT.


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

We have more data than I though, so we can try to increase the size of the model by a decent amount.
In order to keep things behaving well (not overfitting too much), not increase the number of parameters
too much and to ease the training (it takes too much time when doing from scrach) I'm creating a 
depthwise-separable convolution layer to subsitute the convolutional layers with.

Let's see first how the small_lenet works when using depthwise-separable convolutions, just to test.
It erases a lot of parameters, so the overfitting risk becomes a lot lower.

With the depthwise convolutions, the already small lenet, becomes even smaller, going from 20k parameters to 3k,
so it is quite hard for it to be trained considering the data augmentation already in place. So let's keep the
depthwise convolutions for when we find some kind of overfitting and for more complex nets and let's continue with
normal convolution layers for now.

The last linear layers of the resnet are enormous, so I think that instead of decreasing number of layers I should decrease
the depth and, in that way, the amount of channels. Otherwise, 16 M parameters are a ton of then for just 1.8k images. However,
it seems that even just having 1.8k images the model does not overfit too severely and the test gets past 0.8 accuracy super fast.
After that, the models goes straight to overfitting however, and the test loss seems to stabilize and just move arround that point
as well.

Let's try and see what happens if we put the depthwise convolutions here. The number of parameters will not be reduced that
much, but that is because over half the parameters are actually inside the last linear layers. However, let's see first how
does it go with the depthwise convolutions and then let's reduce the channels.