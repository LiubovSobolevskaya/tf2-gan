See [DCGAN](https://arxiv.org/pdf/1511.06434.pdf) for details regarding DCGAN.

## Training       

### lsun/bedroom                                                                                                                 
```bash                                                                                                                 
 python main.py --dataset lsun/bedroom                     
```                                                                                                                      
### cifar10                                                                                                               
```bash                                                                                                                 
 python main.py --dataset cifar10                          
```
### mnist 
``` bash                                                                                                                 
 python main.py --dataset mnist
``` 

## Acknowledgements
I used the following repositories for reference:

*[Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)

*[pytorch examples](https://github.com/pytorch/examples/tree/master/dcgan)

## Results
Gif created from images saved during training.

lsun/bedrooms

![GIF](pics/dcgan_bedrooms.gif)

mnist

![GIF](pics/dcgan_mnist.gif)

cifar10

![GIF](pics/dcgan_cifar10.gif)

These gifs are creates using code taken from 
[DCGAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan#create_a_gif)