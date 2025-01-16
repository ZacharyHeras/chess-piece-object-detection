import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torchvision import transforms as torchtrans
from PIL import Image


# takes the original prediction and the intersection over union threshold
def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')


# plot the image and bounding boxes
# bounding boxes are defined as x-min y-min width height
def plot_image_boundingbox(img, target, classes, class_colors):
    # creates pyplot figure
    fig, a = plt.subplots(1, 1)

    # sets figure size
    fig.set_size_inches(10, 10)

    # displays image on figure
    a.imshow(img)

    # displays bounding boxes on figure
    for i in range(len(target['boxes'])):

        x = target['boxes'][i][0]
        y = target['boxes'][i][1]
        width = target['boxes'][i][2] - target['boxes'][i][0]
        height = target['boxes'][i][3] - target['boxes'][i][1]

        rect = patches.Rectangle((x, y),
                                 width,
                                 height,
                                 linewidth=2,
                                 edgecolor=class_colors[classes.index(classes[target['labels'][i].numpy()])],
                                 facecolor='none')

        a.add_patch(rect)
        a.text(x, y, classes[target['labels'][i].numpy()])

    # displays plot/figure
    plt.show()


if __name__ == '__main__':
    # selects cpu for inference
    device = torch.device('cpu')

    # loads model
    model = torch.load('saved_model/saved_model.pth', map_location=device)

    # define a video capture object
    vid = cv2.VideoCapture(0)

    # defines classes
    classes = [None, 'wpawn', 'bpawn', 'wrook', 'brook', 'wknight', 'bknight', 'wbishop', 'bbishop', 'wqueen', 'bqueen', 'wking', 'bking']

    # defines class colors
    class_colors = [None, 'red', 'blue', 'chocolate', 'greenyellow', 'darkorange', 'indigo', 'yellow', 'green', 'crimson', 'teal', 'gold', 'mediumspringgreen']

    # put the model in evaluation mode
    model.eval()

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        frame = cv2.resize(frame, (480, 480))

        cv2.imwrite("webcam_inference_images/frame.jpg", frame)

        img = Image.open("webcam_inference_images/frame.jpg")

        with torch.no_grad():
            convert_tensor = torchtrans.ToTensor()
            img = convert_tensor(img)
            prediction = model([img.to(device)])[0]

        print('Press q to exit.\n')
        nms_prediction = apply_nms(prediction, iou_thresh=0.01)

        plot_image_boundingbox(torch_to_pil(img), nms_prediction, classes, class_colors)

        # To exit loop press q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
