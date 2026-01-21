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

It seems that using a small resnet with depthwise convolutions achieve a lower performance than using normal convolutions,
but also makes the parameters go from 16M to 9M, from which 8M correspond to the last linear layers, which can be further
decreased by decreasing the number of channels. So it seems that parameterwise, doing dephwise convolutions is much more
efficient. Let's now see the efect of reducing the number of channels instead by adding layers that reduce the number of
channels at the end. We will do the same with the depthwise version.

From the behaviour observed in the WandB graphs, it seems that the depthwise convolution models are more stable in it's
performance, while the losses for the models with the normal convolutions tend to vary a lot more. Also, it seems that
adding the 1x1 convolution to reduce the channels from 512 to 64 reduces by 8M the number of parameters but preserves
most of the performance of the model. Now, let's try putting more layers with a lower amount of channels instead of
the bottleneck at the end.

It seems that adding layers is not really doing much to the performance, also adding depth instead of augmenting
channels reduced the number of parameters. However, it may seem that our model is kind of huge considering that
the slides tell us to use compact models.

Okay, let's now try to achieve similar performances we have until now (between 0.75 and 0.8) with a much lower
number of parameters. It would be nice to keep it arround 100 to 300 thousand or lower. Let's first try with some
tiny mobile net type architecture.

Wow, the architecture with 100k parameters got the same performance than the one with 600k of them. And it seems
it can do better if it's trained for longer, since neither the training or test losses became plain. Also, the
overfitting now is inexistent and if I let it train for more time it may seem that there is no underfitting either.
Let's test that and run it for a really huge amount of epochs for the night. If the model seems to underfit the data
at the end while mantaining a low overfitting, then we may expand the model with more depth layers and add residual
connections for them to be well trained.

Even after 2000 epochs of training, the model seems capable of improving further. However, that kind of slow training
will be too much to later do a sweep on it. Most of the architecture "cannot have" residual connections, since we are
halving the size of the input at almost each step, so let's try to increase the complexity of the model (add layers to it)
to see if that increases how fast it fits the data.

I heard from other students that they are getting 0.8 of accuracy with just 2k parameters, which I find incredibly weird
since they are supposedly not using any kind of knowledge distillation. Another issue is that I setted a learning rate
that its too low, so I will increase it to ease the training in the first epochs. Now, since there are people getting
that accruacy with just 2k parameters I want to do it too, of course. Let's first see if we can get that, and then if we
can improve on this with other means. First of all, let's try to see how the first lenet does with a higher learning rate.
The loss went flat pretty early, so I guess it shouldn't have too much of an impact, but let's see. Probably the training
will become more chaotic as well, but that's something we can try to tune later. If this high learning rate causes more
overfitting to appear, we add 2d dropouts between convolution, and everything solved.

Okay, changing the learning rate actually had an enormous impact, which is something I should had notice before, after all
taking larger steps is needed when training from scratch. If we get to a noisy plain later on, I can think about using
a learning rate scheduler that reduces the learning rate after some number of epochs. Now, we are reaching higher accuracies
than we did before with the normal lenet, however the overfitting is also increasing, so I will be setting dropouts now.

After executing both depthwise and normal lenets, the main problem now is that they tend to underfit, the training seems to
plateu quite fast. I will be adding now a learning rate scheduler to reduce the learning rate when the loss gets into a plain.
For now, we reached almost 0.8 accuracy with just 4k parameters in the depthwise lenet. Let's setup the scheduler and see
if the training gets a little better, but it seems that it will be kind of hard.

It seems that what is actually happening is that test and training accuracies are actually increasing, but very slowly. It
also seems that when reaching arround 0.8 it just cannot handle anymore because of the small amount of parameters. This should
actually improve with knowledge distillation, so I want to try to get similar results with even smaller nets first. Also,
surprisingly the lenet with depthwise convolutions (less parameters) has less underfitting than the one with normal convolutions,
something weird, since it should be able to fit better the one with more parameters. On the other hand, with depthwise convolutions
it can be seen as having extra non-linearities and, thus, some sort of extended expression for this short amount of data I guess.
Well, whatever, let's try and reduce even further the number of parameters. Let's stuck with the depthwise convolutions as well.
Lets aim for 1k parameters now. If training goes too bad, let's create the Knowledge Distillation training and push the boundaries
as much as we can.

Of course, having 1k parameters lead to having arround 0.4 accuracy and plateu there for the remaining epochs. This is probably due
to the absurd reduction I did to the net, which makes it just unable to learn anything useful. I can maybe get to use some bottleneck
to reduce parameters instead. First, let's try my new knowledge distillation method to see if it works and helps the accuracy of the
previous lenets.