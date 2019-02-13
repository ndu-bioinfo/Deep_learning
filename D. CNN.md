#### D. CNN. 
##### Week 10
a) HD image greatly increase the demand of nodes, for example , 1k x 1k x 3 images input size = m x 3mi
     
b) image matrix .dot a kernel or filter - in this case this is a vertical pattern finder

          [[1,0,-1],
           [1,0,-1],
           [1,0,-1]]),
 result in matrix of smaller size. In tf: 
tf.nn.conv2d

c) Other filters available: e.g. Sobel filter 1,2,1 ; Schors filter 3,10,3. You can also train the filter by setting the 3x3 matrix to be W1-W9, then use back prop to find optimal W values for edge detection;
     
d) Padding - n x n image filtered with a f x f filter, yields n-f+1 x n-f+1 dimension output matrix; pixels at the edge were counted less frequent than the ones in the center. To avoid that, you "pad" with 0s outside the border; 

e) Valid convolution - no padding; Same convolution - Pad so the output size is the same as input size;

f) Pad size p = (f-1)/2, f is always an odd number; 

g) Strided convolution - stepping up by S steps instead of 1 step at one time: Output dimension = [(n+2p-f)/s + 1, (n+2p-f)/s + 1]

h) cross-convolution : transpose along bottom left - top right (STD textbook case); In DL should be omitted b/c does not really affect the performance 

i) Convolution on RGB images: image n x n x 3, filter f x f x 3, output n-f+1 x n-f+1 x n_c; n_c is the number of filters, or say channels applied;

j) For a complete convolution, ReLU(output + bias), in this example, the inout image as a[l], while the filter as w[l] * each filter element has its own w value

k) Be careful with the orders of dimensions, as sometimes the notations are ordered differently 

l) In convolutional NN, image metric size decrease while channels increase

m) Pooling layers - max pooling, simply take from each corresponding region the max value; Also, average pooling. Max pooling is used more frequently than average pooling, however average pooling can be useful in very deep NN

n) Max pooling in general should not be used in combination with padding; max polling has no parameters to learn, it's just a fixed function

o) Why convolutional NN? 1. Parameters sharing - a feature detector (filter) useful in one part of the image is probably useful in another part of the image 2. Sparsity of connection - each output value is depend on a small number of inputs

##### Week 11
######  Case studies
a) Classic networks: 

LeNet-5 - To recognize hand written images; ~60K parameters; LeCun et al., 1998, sigmoid was used back then
                      5x5x6 filters, -> avg pooling, then 5x5x16 filter -> avg pooling -> fully connected layers-> One_hot output

AlexNet - Similar to LeNet but much bigger; 60 million parameters; ReLU was used; Multi GPUs used; 227x227x3 input-> 11 x 11 x 96 s=4 filters -> max pooling -> 5 x 5 x 256 SAME conv -> max pool -> 3 x 3  x 384 SAME
conv -> 3 x 3 x 384 SAME conv -> 3 x 3 x 256 SAME conv -> max-pool -> FC -> Softmax 1000


VGG-16 (layers) - Focus on Conv = 3x3 filter, s=1, same Max-pool = 2x2, s=2; 138 M parameters
                        224 x 224 x 3 input -> [Conv x 64] filters x 2 times -> Max-pool x 2 times ->  [Conv x 128] filters x 2 times -> ... -> Max-pool -> FC -> Softmax


b) ResNets - Residual network (152 layers).   Residual block: short cut of a[l] path, a[l+1] = g(z[l+2] +a[l]), a[l] skipping 2 linear steps and inject into deeper network, then the short-cut covered steps are called residual blocks. He et al., 2015. This approach avoids the problem of increasing training errors as layers increase. The reason is the addition of a[l] to the activation function of [l+2] layer allow information to be retained if W[l+2]a[l+2]+b[l+2] -> 0.
    

c) 1x1 convolution:  Not useful for a n x n x 1 data, but for n x n x n_c data, can be used to increase, keep or shrink the # of channels n_c.

d) Inception networks: use >1 different filters and apply to the previous tensor a[l], then stack up all output a[l+1], in this case for both conv and pooling you will have to use padding to keep the dimensions stackable.   

e) bottleneck layer is the layer that has the smallest representation, usually associate with 1x1 conv filter

f) Inception module: Using channel concat to stack tensors together. GoogLeNet

###### Practical advises

g) Transfer learning - Get ride of last output layer, freeze previous layers, and only train the last layer; with pre-trained weights, it's much easier to train your own model.

h) If you have larger training dataset, freeze fewer layers, and training the last few layers

i)  With even larger dataset, you can retrain the NN with the previous weights as initial parametes

j) Data augmentation -  mirroring, random cropping, rotation, shearing are good ways to augment your data for image recognition; color shift time 

k) Two sources of knowledge - labeled data, and hand engineering
     - To improve bench mark

l) Ensembling - train serval networks independently and average their output

j) Multi-crop at test  time - run classifier on multiple versions of test images and average results (more specific for image recognition )
     
#### Week 12 Detection and localization

a) bx, by, bw, bh (center coordinate (x,y), width and height of the box, as in percentage)

b) y = [Pc (object?),bx,by,bw,bh,C1,C2,C3 (classification #)], for background, y = [0,?,?,?,?,?,?,?]

c) For Pc, use logistic reg loss, for b box, use sq mean, for C, use softmax loss

d) Sliding window method: First train the convnet with closely cropped car images, then use small stride to scan through a entire image; start with smaller window and then switch to larger windows. The concept is that if there are cars in the image then certain boxes will catch that. However this sliding window strategy is very computationally expensive.

e) How to implement sliding window strategy -  turning FC layer into conv layer, using a filter that has the same size as the input layer, while the total element numbers are the same. Kind like flatten but with some sort of linear-activation functions;

f) Once training is completed, even though the test data has larger image size, still run it with the same model, and in the final output layer, each 1x1x4 tensor is a representation of the sliding window.

g) Bounding box prediction - Grid the image, then assign object of interest to a specific grid; in the final layer, each of the grid cell has n_C dimensions; In YOLO, bx and by are between 0 and 1, while bh and bw are > 1

h) Intersection over union - compute the size of intersection/size of union; if IOU>= 0.5, then predicted bounding box is OK answer; 

i) Non-max suppression - Object size is larger than grid, and end up with many different detections; suppress those that overlap with lower probabilities; discard box with pc <=0.6, pick the box with the highest pc and discard remaining box with IOU >=0.5

j) Anchor boxes - for overlapping objects; predefine two types of anchor boxes, basically repeat y vectors twice. 

k) YOLO algorithm - For each grid cell, get 2 predicted bounding boxes-> get rid of low prob predicitons -> for each class, use non-max suppression to generate final prediction.   

#### Week 13
###### Face recognition

a) Face verification vs. recognition - verification 1:1 problem, if the person is what the ID shows to be; recognition 1:K problem, output output ID is in the database, or not recognized

b) One shot learning - Usually NN does not work well with one sample; d(img1,img2) = degree of diff -> if d(img1,img2) > tal(critical value) or not 

c) Training of function d - Siamese network: img -> NN -> [128,] vector; d(img1,img2)= || f(img1) - f(img2) ||^2

d) Triplet loss function - 

*  Anchor (in database) <-compare-> img{pos:neg}   ||f(A)-f(P)||^2-||f(A)-f(N)||^2+alpha<=0
*  Given 3 images A, P, N L(A,P,N) = max(||f(A)-f(P)||^2-||f(A)-f(N)||^2+alpha, 0), if not 0, then loss function >0
*  J = sum{i=1->M}L(A(i),P(i),N(i)); for training, multiple positive pictures are required.
*  Choosing the triplets A,P,N: random selection does not work well b/c d(A,P)+alpha<d(A,N) is easily satisfied; for training, choose negative that result in d(A,N) similar to d(A,P);

e) Training set with A + P and N pairs for the training, then compute the values for [128,] vector output, then in application compare this vector to new img processed with the same model

###### Neural style transfer

f) Content (C), Style (S), and -> Generated image (G)

g) J(G) = alpha x J_content(C,G)+beta x J_style(S,G); so the goal is just to reduce J(G) 

h) J_content(C,G) = 1/2 x ||a[l](C) - a[l](G)||^2. What about 1D and 3D data convolution? Using 1D and 3D filter