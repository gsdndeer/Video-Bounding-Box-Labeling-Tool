# Video-Bounding-Box-Labeling-Tool

A video labeling tool using SiamMask tracker, and it can create VOC and YOLO format file.

Originally, SiamMask is a single object tracking framework. However, it is necessary to label multiple bounding boxes at the same time. Therefore, I expand the SiamMask to multi-object tracking.

<img src="https://github.com/gsdndeer/Video-Bounding-Box-Labelling-Tool/blob/master/figures/init.gif">



## Usage

1. Clone the repository
```
git clone https://github.com/gsdndeer/Video-Bounding-Box-Labeling-Tool.git
```

2. Environment setup, following [SiamMask](https://github.com/foolwood/SiamMask#environment-setup)

3. Download the [SiamMask model](https://github.com/foolwood/SiamMask#demo)

4. Put the video under /Video-Bounding-Box-Labeling-Tool-master/data/, and run ```python video_to_image.py``` to produce images.

5. Run ```python labelling.py```
```
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../tools/labelling.py --resume SiamMask_DAVIS.pth --config config_davis.json --base_path ../../data/foldername  
```

### Notes

  * If you want to add new tracker, press ```crtl+c```
  
  <img src="https://github.com/gsdndeer/Video-Bounding-Box-Labelling-Tool/blob/master/figures/add_tracker.gif">
  
## Acknowledgement

1. [SiamMask](https://github.com/foolwood/SiamMask) 
  


