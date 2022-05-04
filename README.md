## Installation 


```bash
   $ git clone https://github.com/aibenStunner/social-distancing-detector.git
   $ cd social-distancing-detector
   $ pipenv shell
   $ pip install -r requirements.txt
   $ python social_distancing_detector.py --input pedestrians.mp4 --output output.avi --display 1 (if you will test the app using a vid)
   $ python social_distancing_detector.py --output output.avi --display 1 (if you will test the app using a webcam/cctv)
```
## Controls

```bash
   (exit = q)
   (pause = p)
   (adjust top down points = 1-4)
   (change topdownview background = v)
   (capture new frozen topdownview background = i)
```

## References

```bash
   (source code)
   $ https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/ 
   $ https://github.com/aibenStunner/social-distancing-detector 

   (top view/bird eye)
   $ https://www.geeksforgeeks.org/perspective-transformation-python-opencv/ 
   $ https://www.youtube.com/watch?v=PtCQH93GucA&t=625s

   (text to speech alarm)
   $ https://towardsdatascience.com/build-a-motion-triggered-alarm-in-5-minutes-342fbe3d5396
```

