Sequential Recommendation with Reinforcement Learning
===============
## Dataset

You must download the raw files of the dataset on data/dataset_name/


## Try it out

Training self-attention based reward model on eachmovie dataset

`python main.py --templates sasreward --dataset_type eachmovie`

Training self-attention based recommendation model on netflix dataset

`python main.py --templates sasrec --dataset_type netflix`