# DINO-pedestrian
Evaluating the DINO object detection model on a custom pedestrian dataset. 

- preprocessing.ipynb takes care of the annotations file. The dataset is loaded and then converted to a pandas dataframe for easier analysis. By merging the information about the image and its constituent bounding boxes, I visualised the initial bounding boxes on respective images. Like so -
![image](https://github.com/user-attachments/assets/d411e6c5-c345-4493-b7ee-1019cac8f0d4)

Then this dataframe is sampled randomly, to facilitate the splitting. And then 160 images make the training set. And 40 images make the validation set. New json files are created for both sets' annotations.  

- dinoonpedestrian.ipynb was trained on Kaggle with GPU T4x2 accelerators. I cloned the DINO repository and installed the dependencies. When evaluated on pedestrian data and the checkpoint path of 4scale DINO, I received the following metrics -


     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.486
  
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.823
  
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.509
  
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.372
  
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.599
  
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697
  
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.108
  
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.506
  
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.602
  
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.545
  
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.663
  
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.750
 

The visulisation after eval resulted in examples like these - 

![image](https://github.com/user-attachments/assets/9ba4eba6-27c6-4003-b22c-ede4a7777ef6)
![image](https://github.com/user-attachments/assets/3e2cbd79-4eaf-4f1e-a044-8968d672ebd2)

Since the number of classes is 91 in DINO, it picked up objects like 'chair', and 'car' too. But we expect after fine-tuning, it will only recognise 'pedestrian'.

![image](https://github.com/user-attachments/assets/e4b99dbc-151c-4c1c-879b-e7458858c5eb)


