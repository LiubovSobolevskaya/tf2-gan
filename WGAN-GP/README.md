See [WGAN-GP](https://arxiv.org/pdf/1704.00028.pdf) for details regarding WGAN-GP.

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
* [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training)

## Results
Gifs created from images saved during training:

lsun/bedrooms

![GIF](pics/wgan-gp_lsun.gif)

cifar10

![GIF](pics/wgan-gp_cifar10.gif)

mnist

![GIF](pics/wgan-gp_mnist.gif)

These gifs are creates using code taken from 
* [DCGAN Tutorial] (https://www.tensorflow.org/tutorials/generative/dcgan#create_a_gif)