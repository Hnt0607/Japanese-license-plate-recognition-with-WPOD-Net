
import cv2
import numpy as np
from lib_detection import load_model, detect_lp
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
# from keras.models import load_model




# CODE----------------------------------------------------------------



wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)


def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


def get_plate(image_path, Dmax=608, Dmin = 608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, lp_type = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    # cv2.imshow("Detected License Plate", LpImg[0])  # Displaying the first license plate image in the array
    # cv2.waitKey(0)  # Wait for a key press to close the window
    return vehicle, LpImg


def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction



def detect_recognize_plate(ImgPath):
    image_path = ImgPath
    vehicle, LpImg = get_plate(image_path) 
    # cv2.imshow("Anh bien so", vehicle)
    # cv2.waitKey()
    # CROP IMAGE
    image_width, image_height = LpImg[0].shape[1], LpImg[0].shape[0]
    top_margin = int(image_height * 0.36)
    left_margin = int(image_width * 0.19)
    cropped_image = LpImg[0][top_margin:, left_margin:]

    #Resize cropped image
    desired_width = 154
    desired_height = 95

    resized_image = cv2.resize(cropped_image, (desired_width, desired_height))
    cv2.imshow("Anh bien so sau margin", LpImg[0])
    cv2.waitKey()
    if (len(cropped_image)):
        # plate_image = cv2.convertScaleAbs(cropped_image, alpha=(255.0))
        plate_image = cv2.convertScaleAbs(resized_image, alpha=(255.0))

        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(7,7),0)
        binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn_image = resized_image.copy()
    # LOOP CHECK NUM_CHAR < 4
    num_rate = 2.5
    num_char = 0
    while (num_char < 4):
        test_roi = plate_image.copy()
        test_image = plate_image.copy()
        crop_characters = []
        temp_characters = []
        digit_w, digit_h = 30, 60
        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 0.9<=ratio<=num_rate:
                if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                    # Draw bounding box arroung digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                    # Sperate number and gibe prediction
                    curr_num = thre_mor[y:y+h,x:x+w]
                    test = test_image[y:y+h,x:x+w]
                    temp_characters.append(test)
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
                    cv2.rectangle(drawn_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Count character in crop character and add num_rate
        num_char = len(crop_characters)
        num_rate += 0.25
        if (num_rate >= 10): break
    #limited to 4 recognized characters
    if len(crop_characters) > 4:
        crop_characters = crop_characters[:4]
    print("Detect {} letters...".format(len(crop_characters)))

    json_file = open('data/Character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("data/digit_recognition_model.h5")
    print("[INFO] Model loaded successfully...")
    labels = LabelEncoder()
    labels.classes_ = np.load('data/license_character_classes.npy')
    print("[INFO] Labels loaded successfully...")
    
    

    final_string = ''
    for i,character in enumerate(crop_characters):
        title = np.array2string(predict_from_model(character,model,labels))
        final_string+=title.strip("'[]")
        
        # cv2.imshow(f"Character {i+1}", character)
        # cv2.waitKey()
    
    while (len(final_string) < 4):
        final_string = "•" + final_string
    path_vehicle = "static/vehicle_image.jpg"
    path_drawn = "static/drawn_image.jpg"

    vehicle_image_rgb = cv2.convertScaleAbs(vehicle, alpha=(255.0))
    vehicle_image_rgb = cv2.cvtColor(vehicle_image_rgb, cv2.COLOR_BGR2RGB)


    drawn_image_rgb = cv2.convertScaleAbs(drawn_image, alpha=(255.0))
    drawn_image_rgb = cv2.cvtColor(drawn_image_rgb, cv2.COLOR_BGR2RGB)
    
    vehicle = cv2.resize(vehicle,(224, 150))
    cv2.imwrite(path_vehicle, vehicle_image_rgb)
    cv2.imwrite(path_drawn, drawn_image_rgb)
    # cv2.imshow("Cac ky tu da cat", drawn_image_rgb)
    # cv2.waitKey()
    return path_vehicle, final_string, path_drawn


# IMGPath = './images/car/19.jpg'
# detect_recognize_plate(IMGPath)
# for i in range(1, 10):
#     detect_recognize_plate(f"./images/car/0{i}.jpg")

# for i in range(10, 23):
#     detect_recognize_plate(f"./images/car/{i}.jpg")