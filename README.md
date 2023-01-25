# Code and functions for analysis of multi-site local field potentials

## Functions for processing neural and behavioral signals


The use of multimodal neural signals to predict behavior and pathologies in preclinical models has gained interest in the recent years given the advance of techniques for big data analysis. While growing availability of machine learning packages promises an exiting future to the field, pre-processing of the data like artifacts removal and aligment of neural and behavioral signal, may take considerable amount of time and are subject to errors. While commercial solutions exist, they are generally expesive and don't allow for processive analysis. This module aims to contribute to the advance of systems neuroscience by providing functions for processing neural and behavioral signals. I hope this will save time, increase reproducibility between studies and reduce clicking!

A module with the functions currently available can be found [here](https://github.com/casey-e/Multi-site_LFP/blob/main/Functions/Processsing_functions.py)  

## Example

In the example provided, a mouse was videotracked an over-night while explored a novel cage with a feeder at the middle of the right located wall of the cage. The file  [Centroids.csv](https://github.com/casey-e/Multi-site_LFP/blob/main/Functions/Example/Centroids.csv) contains "X" and "Y" positions at each timestamp, obtained from the videotrack, and the file [feeding.pickle](https://github.com/casey-e/Multi-site_LFP/blob/main/Functions/Example/feeding.pickle) contains timestamps corresponding to the removal of a food pellet from the feeder (recorded as a TTL sent by the feeder).

The example use functions provided [here](https://github.com/casey-e/Multi-site_LFP/blob/main/Functions/Processsing_functions.py) to scale X and Y values according to the dimensions of the cage used and calculate the scalar speed. Then it identifies periods of rest initiations (rest_starts), identified as stop of locomotion (yes, this is not perfect) and uses a function to identify the positions of rest initiations and feeding. Hovever, positions of rest initiations largely overlap with positions of feeding (figures below).  
  
  
![image](https://user-images.githubusercontent.com/92745842/214613390-0dfdf28b-ee0c-409e-8a6a-90e35b4c4c23.png)
![image](https://user-images.githubusercontent.com/92745842/214613427-86663bdd-a95d-4f8f-a5b9-486cc94c8a8b.png)  
  
  
Whith the function peri-events histograms, it is clear that the mouse stop moving to eat, which is being confused with rest.  
    
    
  ![image](https://user-images.githubusercontent.com/92745842/214614241-b2e37c54-2593-45cc-9ed6-2f4a93d8c1eb.png)  
    
 The function clasify_events_base_on_time allows for a binary classification of the timestamps of an event, based on its temporal proximity with another event. This allows to differentiate real rest from feeding (false rest).  
   
![image](https://user-images.githubusercontent.com/92745842/214615462-ae5a1da5-cb96-4a29-bb15-bffec1d0c922.png)
![image](https://user-images.githubusercontent.com/92745842/214615490-6176a0e1-6e54-441c-830a-97652deabe66.png)
![image](https://user-images.githubusercontent.com/92745842/214615514-4728735e-61c6-4012-9315-299869bb77ad.png)
  
  ![image](https://user-images.githubusercontent.com/92745842/214615623-34b847dc-175a-4286-8d70-2ec14c660882.png)
  
  
We can further confirm this by calculating the distance to the most likely feeding position at each time and pltting peri-event histograms.  
  
    
    
  ![image](https://user-images.githubusercontent.com/92745842/214616139-b01a71a9-b8e4-49bc-ad30-3339d3eb2f00.png)


