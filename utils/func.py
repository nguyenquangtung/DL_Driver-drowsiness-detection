def get_results(results):
    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
    lst_class = results[0].boxes.cls.cpu().tolist()
    return boxes_xyxy, lst_class


def crop_image(frame, bbox):
    """
    Crop the input frame using provided coordinates and save the cropped image.

    Parameters:
        input_image_path (str): Path to the input image file.
        output_image_path (str): Path to save the cropped image file.
        coords (tuple): Coordinates of the form (x1, y1, x2, y2) specifying the region to crop.
    """
    # img = cv2.imread(frame)
    bbox = bbox.astype(int)
    x1, y1, x2, y2 = bbox
    cropped_img = frame[y1:y2, x1:x2]
    # Crop the image using the provided coordinates
    return cropped_img


def is_inside(box1, box2):
    if box1 is not None and box2 is not None:
        # Kiểm tra tọa độ x và y
        if (
            box1[0] >= box2[0]
            and box1[2] <= box2[2]
            and box1[1] >= box2[1]
            and box1[3] <= box2[3]
        ):
            return True
    return False


def calculate_area(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    return area
