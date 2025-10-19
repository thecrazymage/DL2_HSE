# Diffusion models 1

Lecturer and seminarian: [Denis Rakitin](https://www.hse.ru/org/persons/190910999/)

Recordings (in Russian): [lecture](), [seminar]().

## Annotation

Tomorrow's lecture and seminar will be devoted to an introduction to diffusion models. Diffusion models are currently the most popular approach to generative modeling due to their high-quality generation and diversity (mode coverage) of the learned distribution. The idea behind diffusion models is to consider the process of gradually transforming data into pure noise and construct its inverse in time, which will transform noise into data. In the lecture and seminar, we will work with noise processes and derive the classic DDPM model, which proposes to minimize the KL-divergence between the “true” reverse process that converts noise into data and the denoising process specified by the neural network. In the process, we will see that this procedure is equivalent to training a denoiser neural network that predicts a clean object from a noisy one. In addition, we will interpret the resulting denoising process: in it, each step corresponds to replacing part of the current noisy image with an (increasingly high-quality) prediction of the denoiser.