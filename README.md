# Chess Piece Detection Project

This mini-project consists of three main files:

1. **finetune_faster_rcnn.py**:  
   This script is used to train a model for chess piece detection. It fine-tunes a pre-trained Faster R-CNN model to detect chess pieces in images.

2. **inference.py**:  
   This script tests the trained model by running inference on a set of images to identify and classify chess pieces.

3. **webcam_inferences.py**:  
   This script captures images from a webcam and uses the trained model to identify and classify chess pieces in real-time.

## Setup Instructions

To use this repository, you must first create a virtual environment using Python 3.10.12. Then, install the required dependencies from the `requirements.txt` file.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate.bat`
pip install -r requirements.txt
```

## Old Project Notice
This project is older and is provided for reference purposes. It is not actively maintained, and was last updated in 2022.

## Output
Below is an sample output showing the model detecting and classifying chess pieces:

![](https://github.com/ZacharyHeras/chess-piece-object-detection/blob/main/output.png)

## Primary References

This project is based on the following key references:

1. **OpenCV Documentation**:  
   [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

2. **PyTorch Vision Detection Code Reference**:  
   [PyTorch Vision Detection](https://github.com/pytorch/vision/tree/main/references/detection)

`coco_eval.py`, `coco_utils.py`, `engine.py`, `transforms.py`, and `utils.py`, are directly from the Pytorch's reference listed above.
