# Work to do

## Experiments

### First without data augmentation !!!

### MLP on small images: end to end vs svm

STATEMENT: Redefine the structure of the newtork (adding new layers, changing image size, ...)

1. Train SimpleModel once (use make_like_simple)
2. Train SimpleModel with more hidden width [512, 1024, 2048, more?]. Select best width.
3. DO THE REPORT OF THE PREVIOUS EXPERIMENTS!!!
4. Train SimpleModel with more hidden layers (300 hidden width) [from 1 to 4]. Select best depth.
5. DO THE REPORT OF THE PREVIOUS EXPERIMENT!!!
6. Test different image sizes (32, 64, 128, 224, 256). Select best image size.
7. DO THE REPORT OF THE PREVIOUS EXPERIMENTS!!!
8. If we find overfit -> data augmentation

STATEMENT: Extract the output from a given layer as descriptor and apply svm on it

9. Choose best model above, extract descriptors and use SVC ffor classification. Best SVC from last week.
10. DO THE REPORT OF THE PREVIOUS EXPERIMENT!!!

### MLP as dense descriptor: end to end vs BoW

STATEMENT: Divide the image into small patches
STATEMENT: Extract the prediction for each patch, and agregate the final prediction (end to end)

1. Test different patch sizes (4, 8, 16) [descriptor size fixed at 128 like SIFT]. Choose best patch size.
2. DO THE REPORT OF THE PREVIOUS EXPERIMENT!!!
3. Test different descriptor sizes [32, 64, 128, 256]
4. DO THE REPORT OF THE PREVIOUS EXPERIMENT!!!

STATEMENT: Take each patch output, given a layer, as a dense descriptor and apply Bow

3. Extract descriptors for each image patch and do BoVW. -> KMeans + histograms + SVC (Best parameters from last week)

4. finish the report!!!
