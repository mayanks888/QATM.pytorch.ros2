import numpy as np
import os
import cv2
import keras
#############################################33333

def color_Detect():
    model = keras.models.load_model("/home/mayank_sati/pycharm_projects/tensorflow/traffic_light_detection_classification-master/traffic_light_classification/saved_models/model_b_lr-4_ep50_ba32.h")

    traffic_light=["green", "red", "black"]
    # traffic_light=["red", "green", "black"]

    input_folder="/home/mayank_sati/Desktop/one_Shot_learning/col"
    # input_folder="/home/mayank_sati/Desktop/traffic_light/sorting_light/small data"
    # for root, _, filenames in os.walk(input_folder):
    #     if (len(filenames) == 0):
    #         print("Input folder is empty")
    #     # time_start = time.time()
    #     for filename in filenames:
    filename = '/home/mayank_sati/Desktop/git/2/AI/QATM/data/cust_template/myimage_outPut.jpg'
    # filename = '/home/mayank_sati/Desktop/git/2/AI/QATM/data/cust_template/myimage.jpg'
    test_image = keras.preprocessing.image.load_img(path=filename, target_size=(32, 32))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255
    result = model.predict(test_image)
    print(result)
    light_color = traffic_light[result.argmax()]
    print(light_color)
#######################################################
    # image_scale = cv2.imread(filename, 1)
    # cv2.putText(image_scale, light_color, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 255, 0),
    #             lineType=cv2.LINE_AA)
    # cv2.imshow('streched_image', image_scale)
    #
    # ch = cv2.waitKey(500)  # refresh after 1 milisecong
    # if ch & 0XFF == ord('q'):
    #     cv2.destroyAllWindows()
    # cv2.destroyAllWindows()
    return light_color

if __name__ == '__main__':
    color_Detect()