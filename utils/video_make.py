import cv2, glob
import natsort
import pandas as pd
import numpy as np

pred_labels_path = "/project/CollabRoboGroup/msy9an/GoalPrediction/src/predictions.csv"
predicted_labels = pd.read_csv(pred_labels_path, index_col=0)
predicted_labels = np.asarray(predicted_labels).reshape(-1,2)

activities =   ["Push rotor into housing ", "Align stator with housing", "Tighten bolts using ratchet", "Final Inspection"]

data_dir = "/project/CollabRoboGroup/msy9an/GoalPrediction/outputs_cccamdemo/rgb_images/color*.png"
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640+640,480+40))

font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (30*4, 30)
  
# fontScale
fontScale = 0.8
   
# Blue color in BGR
color = (0, 0, 0)
# Line thickness of 2 px
thickness = 2
count = 0
for image in natsort.natsorted(glob.glob(data_dir)):
    # print (image)
    img1 = cv2.imread(image)
    depth_file = image.replace("rgb_images", "depth_images").replace("color", "depth")
    
    # Read Second Image
    # print (depth_file)
    img2 = cv2.imread(depth_file)
    
    # print (img2.shape)
    img3 = cv2.imread(image.replace("rgb_images", "keypoints"))# .replace(".png", ".png.png"))
    
    img3 = cv2.resize(img3, (640, 480), interpolation = cv2.INTER_AREA)
    # concatanate image Horizontally
    # print (img1.shape, img2.shape, img3.shape)
    merged_image = np.concatenate((img1, img3), axis=1)
    
    pad = cv2.copyMakeBorder(merged_image, 40,0,0,0,  cv2.BORDER_CONSTANT, value=(255, 255, 255))   
    # print (pad.shape)
    image = cv2.putText(pad, 'Predicted: {}     ,     Actual: {}'.format(activities[int(predicted_labels[count,0])], activities[int(predicted_labels[count,1])]), org, font, fontScale, color, thickness, cv2.LINE_AA)
    if count >15:
        print (count)
        out.write(pad)
    count += 1
    
    if count >= predicted_labels.shape[0]-30:
        break
    
    
out.release()
cv2.destroyAllWindows()
