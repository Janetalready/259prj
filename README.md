# 259prj
```
FinalProject
|
|--- raw_data3 
|     |--- watch
|     |--- play
|     |--- search
|
|--- data_generator.py(manually generate data) 
|--- data_preprocessing.py(preprocess data to get action sequence and its label)
|--- classifier.py(bow model and glove model)
|--- visualization.py(visualiza experimental results)
```

The google drive link of data: https://drive.google.com/file/d/1lK-uFZdCoNi0dEOYC5JJn7V2Gi1pMhWE/view?usp=sharing 
Before run, download and drag all dataset and model file into the 'raw_data3'

The workflow is
```
      Data preprocessing(run data_preprocessing.py) 
  --> Train classifier(run classifier.py with bow part for bow+bayes in main function, or run classifier.py with glove_train part to train the glove embedding, or run classifier.py with glove+CNN part to train the CNN classifier) 
  --> visualization(run visualization.py with its different methods to visualize the experimental results)
```





