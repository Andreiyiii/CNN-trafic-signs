import cv2
import numpy as np
import os
import sys
import tensorflow as tf


from sklearn.model_selection import train_test_split
ALL_SIGNS = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}
IMG_WIDTH = 30
IMG_HEIGHT = 30
model=tf.keras.models.load_model("3conv1pool.h5")

def main():

    if len(sys.argv)!=2:
        sys.exit("Usage: python checker.py image.png")
    image_path= sys.argv[1]
    img=cv2.imread(image_path)
    img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
    img = np.expand_dims(img, axis=0)  # now shape is (1, 30, 30, 3)

    prediction=model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    print(f"Predicted class: {ALL_SIGNS[predicted_class]}")

if __name__ == "__main__":
    main()