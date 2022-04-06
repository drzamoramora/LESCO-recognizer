# LESCO-recognizer

## Summary:

This project is an evolution of Juan Zamora's Doctoral dissertation: "Video-Based Costa Rican Sign Language Recognition for Emergency Situations" from Aspen University. The main idea is to create a recognition framework that is able to "understand" LESCO from videos and be able to translate each video into a set of keywords in spanish (or any language) for full video-2-text translation.

## Dataset:

The dataset has been uploaded to [Zenodo](https://zenodo.org/record/6345338#.Yi-HZXrMKUk)

The dataset is composed of 39 signs. There are three videos for each sign on each folder. Videos have been cropped and are on average 1 second long.  This dataset contains a total of 117 videos and 20 additional videos of LESCO sentences. 

## Methodology: 

Design Science (DS) has been used as the main research methodology. DS provides a set of practices to translate an idea into a product leveraged into a set of iterations where every artifact is tested and evaluated for further iterations. Each iteration could be seen as an "Agile" iteration where ne wideas and hypothesis are tested.

## Itertations

- Iteration 1: translate LESCO videos into text by using similary measures
- Iteration 2: translate LESCO videos into text by using Deep Learning
- Iteration 3: evolution of Iteration 1 with other dimensional reduction algoritms.
- Iteration 4: cherry-picked frames for each video were selected to reduce the amount fo data for training: the hypothesis is that key frames that show relevant hand shapes are sufficient for sign recognition.

## Acknowledgements

- This project has been actively supported with LESCO translators, and research guidance over inclusive technologies by IncluTec from the Intituto Tecnologico de Costa Rica. 
