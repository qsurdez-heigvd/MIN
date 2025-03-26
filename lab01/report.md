# Answer

Author: Quentin Surdez

## Table of contents

- [DCGAN](#dcgan)
    - [Question 1.1](#question-11)
    - [Question 1.2](#question-12)
    - [Question 1.3](#question-13)
- [DCGAN - Outputs](#dcgan---outputs)
    - [Question 1.4](#question-14)
- [Pix2Pix](#pix2pix)
    - [Question 2.1](#question-21)
    - [Question 2.2](#question-22)
    - [Question 2.3](#question-23)
    - [Question 2.4](#question-24)
    - [Question 2.5](#question-25)
    - [Question 2.6](#question-26)

## DCGAN

### QUESTION 11

> ❓ What makes this practical work experiment a self-supervised one?

This practical work will transform the raw data without using explicit labels, creating a representation learning task where the model will have to learn meaningful pattern from the data itself.

The emphasis is on the fact that none of the data will be labeled and that, using GAN, it will learn by itself from the data.

### QUESTION 1.2

> ❓ Please look at the generator architecture. How many time do we multiply by two the size of the input and what are the layers responsible for this? ❓ What is the use of the Reshape layer in this code?

The generator increases the space dimensions of the input twice. This happens through the `Conv2DTranspose` layers. We can observe that initially our tensor is 7x7x256, the first `Conv2DTranspose` transforms it into a 14x14x256 tensor. It is thanks to the `strides=2` parameter. Then the same transformation happens with the second `Conv2DTranspose` where our 14x14x256 tensor transforms into a 28x28x256 tensor.

The `Reshape` layer serves as some kind of restructuration from the dense 7x7x256 flat vector (dense layer) into a 3D tensor with spatial dimensions of 7x7 and 256 channels (convolutional layer). The will allow the `Conv2DTranspose` layers to work properlly and is the starting point for the upsampling of the images.

### QUESTION 1.3

> ❓ Please look at the discriminator architecture, notices that it's a CNN classifier (between fake and real images). If you where to classify RGB images of multiple animal classes (cats, dogs and ducks for example), what would you need to change?

We're starting with a discriminator designed for binary classification of grayscale MNIST images (28x28x1). To modify this architecture for multi-class RGB animal classifications, we need some changes:

1. INPUT: We need to change the input change, from grayscale to RGB (widthxheightx3)
    
2. OUTPUT: Modify from binary to multi-class. The use of the softmax activation function is better to normalize across multiple classes and represent a probability distribution
    

These are the two main changes I see we should do to handle RGB images of multiple classes. We could also consider increasing the size of the neural network as well.

## DCGAN - OUTPUTS

### QUESTION 1.4

> a❓ Can you rely only on the loss of the generator and discriminator to choose the best model? If no, provide a counter-example.
> 
> b❓ In the third experiment we significatively reduced the number of parameters of the discriminator compared to the other experiments. Did it helped the generator to produce better images? Why?
> 
> c❓ Compare experiments 1 and 6. Remember, They use the same number of filters but have a different architecture.
> 
> d❓ In experiment 5, we decrease the number of paramers of the generator. What was the impact?

**a**: No, relying solely on the generator and discriminator loss values is insufficient for choosing the best DCGAN model. These metrics can be misleading for several fundamental reasons:

The loss function is not directly related to the generateed image quality. Other metrics can be put used to measure the quality of a producede image such as the Inception Score introduced by Salimans et al. (2016).

The best counter-example that comes to mind would be model collapsing where the generator begins to create only a very specific subset lacking in diversity of images and the discriminator get optimized towards this specific subset.

**b**: The first thing we observe are the loss functions. They are quite balanced and the relationship between the generator loss and the discriminator loss is stable. We also observe that the volatility of the loss function is quite low.

Watching the images of the third model, we can see that they are pixelized. Or at least, more pixelized than the ones from the others models. The shoe shape is well defined, but the maybe sweater shape is not well defined.

I don't think it helped the generator proudce better images compared to the other models where the images are less pixelized and better defined. However, we can see that there is less volatibility in the loss functions. Maybe with more epochs this model could create better images as the signals from the discriminator are more consistent. This is deduced by the lack of volatibility between the two loss functions.

**c**: Experiment 1 shows a gradually diverging pattern where:

- Generator loss steadily increases from 0.75 to 1.0
- Discriminator loss gradually decreases from 0.7 to 0.6
- The divergence increases over time, suggesting growing discriminator dominance
- Implements LeakyReLU activation function
- Contains batch normalization layers

Experiment 6 demonstrates:

- Higher initial volatility with pronounced loss spikes
- Generator loss that eventually trends upward but with more fluctuation
- A similar final discriminator loss (~0.57) to Experiment 1
- Uses standard ReLU
- Omits batch normalization

The clear difference is at the start of the training where the loss function makes a lot of spikes for experiment 6 and is quite smoother in experiment 1. However, despite these differences, the two models seem to converge to the same value of loss functions.

**d**: The generator loss function is, at first, very high (~6) and the discriminator loss function remains quite low (~0.1) throughout training.

Then a gap begins to form between the discriminator and the generator that cannot be bridged. In deed, the generator having too few convulational layers, it will impact its capabilities for capturing spatial correlations and fine details for convincing image generation. Regardless of training duration.

## Pix2Pix

### QUESTION 2.1

> ❓ We want to colorize grayscale images. Is there only > one valid colorized output?

No there are many. RGB is the most well-known color space, but other could be more beneficial to us, or less. For example there is the CIELUV color space that is based on human perception.

### QUESTION 2.2

> ❓ Pix2Pix is a Conditional Adversarial Network, in the practical work with DCGAN we were using noise as input. What is different in the input we have here?
> 
> ❓ How does our colorization task relates to a problem where we would want to take photos as inputs and make them look like paintings? Please provide another task that would be related to these problems.

As written un the introduction of the Pix2Pix paper, we understand that, now, the input is an image. It's no longer a vector of noise, but a complex and structured image. Thus, there is an influence of the input on the result contrary to DCGANs where there was no influence at all as it was a vector of noise generally following a normal distribution.

Both colorization of grayscale image and photos to paintings are examples of image-to-image transformation tasks. In the two tasks we want to take an image and tranform it while retaining its structure.

Examples of other image-to-image transformation tasks would be a night-to-day image transformation or an aerial-to-map image transformation.

### QUESTION 2.3

> ❓ Why do you think the model predictions look like this? In which way a Pix2Pix GAN would be useful to improve the results?

We can observe that the image predicted by the model all look brownish and desaturized.

The results of the image prediction are influenced by the loss function. The MSE (Mean-Squared Error) is a very good function to optimize towards the average of all possible colorizations in our case. Thus, the images are brownish and desaturized because this is the result of the average of color throughout the dataset.

What we have to keep in mind with MSE is that being "somewhat wrong" across all pixels is better for it than being really wrong on some pixels but correct on others.

There are many reasons why a Pix2Pix GAN would be more useful:

1. Adversarial loss: Forces the generator to produce realistic, vibrant colorizations.
    
2. Perceptual quality over pixel-wise accuracy: The discriminator evaluates overall realism rather than pixel-wise accuracy.
    
3. Image generation: The generator aims to create real images or at least, images that would be categorized as real by the discriminator, rather than valuing the pixel-value accuracy.
    

For all the above reasons, the Pix2Pix would be helpful to generate better colorizations of grayscale images.

### QUESTION 2.4

> ❓ What is the advantage to use L\*A\*B\* color space instead of RGB in our case?

L\*A\*B\* color space is intended as a perceptually uniform space, where a given numerical change corresponds to a similar perceived change in color.

This color space is quite interesting for our problem, because the grayscale image only has value for the L\* parameters, thus our model only has to predict the 2 others A\* and B\*.

Another interest of this color space is that it's already normalized between -1 and +1 in all dimensions which makes it quite well suited for machine learning.

### QUESTION 2.5

> a❓ What does it mean to have an input shape of (None, None, 1)?
> 
> b❓ Look at the architecture plot. What are the connections between some layers of the downsampling and upsampling parts?
> 
> c❓ Why do we have only two outputs channels? What is the model output?

[Ref](https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc) for input shape documentation

**a**: An input shape of `(None, None, 1)` gives a lot of informations:

- The first `None` represents the batch size. Its value being None tells the model to process any number of images at once.
    
- The second `None` represents both the width and height of the images. Its value being None tells the model to accept images of any spatial dimensions.
    
- The thirs dimension being 1 means that there is only one channel as expected for grayscale images.
    

**b**: These connections are the disctinctive feature of the U-Net architecture. They allow the model to combine high-level information from the deepest layers to fine details from the earlier stages that may have been lost in the downsampling.

**c**: We only have two output channels because the model is working in L\*A\*B\* color space. As said before, the image being in grayscale means that the L\* parameters does not need to be touched. The model will only work with the A\* and B\* parameters which is why we only have an output layer with only two channels.

These two combined with the initial L\* parameter will be able to be converted back to RGB for example to display the resulting image.



### QUESTION 2.6

> a❓ In the training process, do the discriminator compare pairs of target and predicted images?
> 
> b❓ Let's consider that the generator model is better than the one trained here. Is it probable for the generator to produce an image that is the same as one from the targets set (with the real colors)? Why so?
> 
> c❓ Look at the training code, what is the value we expect the discriminator to give us when the image is fake and the one when the image is real.
> 
> d❓ Provide three colorized images with the model that you find interesting (e.g. well colorized, artistic, disastrous result, ...).
> 
> e❓ Provide an image you have in grayscale (convert one in graycale if you don't have any) and apply the model on your image.

**a**: Yes, thee discriminator does compare pairs of images. It is given both the real both the real target image and the generated one. The loss function is computed with the loss of the two outputs from the discriminator.

We can understand where the conditional aspect of this GAN comes from as it does not only evaluate the generated output but also its relationship with the expected output.


**b**: It is quite unlikely for the generator to create an image with the exact same colours as the ground truth image. In deed, there are some limitations inherent to the problem itself.

- Many colorization can be valid for a specific object, without additional information it would be very difficult if not impossible to produce the same color as the ground truth image. For example, a can of soda may be one color in one country and another one abroad. Both outputs are valid.

However, with a very well trained model, it is possible to create an image that would be plausible and realistic as it is the purpose of the model. Not replicating the ground truth image, but creating an image that would be realistic.

**c**: Below the answer to the question

- The value we expect the discriminator to give us when the image is real is 0.9
- The value we expeect the discriminator to give us when the image is fake is 0. (around 0 is also fine)

**d**:

**e**:
