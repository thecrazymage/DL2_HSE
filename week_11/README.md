# Neural Recommender Systems

Lecturer: [Kirill Khrylchenko](https://www.hse.ru/org/persons/1085421573/)

Seminarian: [Artem Matveev](https://www.linkedin.com/in/artem-matveev-7b2725255/)

Recordings (in Russian): [lecture](https://disk.yandex.ru/d/mzXlT0U3MzEZkQ/%D0%9F%D0%9C%D0%98/DL%202/lecture_11.mp4), [seminar](https://disk.yandex.ru/d/mzXlT0U3MzEZkQ/%D0%9F%D0%9C%D0%98/DL%202/seminar_11.mp4).

## Annotation

Modern recommender systems also make heavy use of deep learning — both discriminative and generative models. A system has to understand content (tracks, products, videos, etc.), model user behavior (short- and long-term preferences), and predict which content each user is likely to enjoy. At the same time, recommender systems come with their own challenges: cold start for new items and users, huge billion-scale item catalogs with heavy-tailed popularity distributions (popularity bias), and constant distribution drift, where all the underlying distributions keep changing along with the rest of the world. 

We will discuss: 
- Two-tower neural networks for candidate generation;
- Sequential recommendation and more modern approaches to generative modeling of users; 
- Semantic IDs and tuning LLMs for recommendation;
- What makes the recommendation domain different from other deep learning domains.