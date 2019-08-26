# Video-Bounding-Box-Labelling-Tool

This tool is using SiamMask tracker. I expand the SiamMask to multi-object tracking. It can create VOC and YOLO format file.

<img src=>



## Usage

1. Clone the repository
```
```

2. Environment setup, following [SiamMask](https://github.com/foolwood/SiamMask#environment-setup)

3. Download the [SiamMask model](https://github.com/foolwood/SiamMask#demo)

4. Put the video under /Video-Bounding-Box-Labelling-Tool-master/data/, and run ```python video_to_image.py``` to produce images.

5. Run ```python labelling.py```
```
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../tools/labelling.py --resume SiamMask_DAVIS.pth --config config_davis.json --base_path ../../data/foldername  
```

### Notes

  * If you want to add new trackers, press ```crtl+c```
  
  <img src=>
  
## Acknowledgement

1. [SiamMask](https://github.com/foolwood/SiamMask) 
  


