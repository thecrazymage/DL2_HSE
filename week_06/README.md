# Segmentation and Detection 2

Lecturer: [Sergey Zagoruyko](https://szagoruyko.github.io/)

Seminarian: [Eva Neudachina](https://www.hse.ru/org/persons/401628708/)

Recordings (in Russian): [lecture](), [seminar]().

## Annotation

**Lecture:**  
In this lecture, we will continue to talk in more detail about segmentation and detection. The focus is split approximately 70% on object detection and 30% on how detection paradigms translate into and power segmentation methods, with a particular emphasis on modern transformer-based architectures.

**Seminar:**  
This seminar gives a hands-on introduction to getting segmentation masks from the attention maps of Transformer models like DETR and SAM. We’ll look at how to use a `forward_hook` and attention proccessors replacement to access and understand attention scores, and discuss ideas from a recent хFlux-based paper](https://arxiv.org/pdf/2502.04320v1). You’ll learn how to turn raw attention maps into usable masks through simple code examples and experiments.