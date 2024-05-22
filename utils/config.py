from pydantic import BaseSettings


class ProjectConfig(BaseSettings):
    # Size frame
    WIDTH: int = 1280
    HEIGHT: int = 720
    SIZE: tuple = (1280, 720)

    # Some basic color
    BLUE: tuple = (0, 0, 255)
    GREEN: tuple = (0, 255, 0)
    RED: tuple = (255, 0, 0)
    WHITE: tuple = (230, 230, 230)

    TIME_THRESHOLD: int = 4
    ALARM_SOUND: str = r"public\alarm.mp3"

    FACE_CASCADE_PATH: str = r"model\haarcascade\haarcascade_frontalface_default.xml"
    LEFT_EYE_CASCADE_PATH: str = r"model\haarcascade\haarcascade_lefteye_2splits.xml"
    RIGHT_EYE_CASCADE_PATH: str = r"model\haarcascade\haarcascade_righteye_2splits.xml"
    YOLO_MODEL_PATH: str = r"model\yolo_model.pt"

    MODEL_PATH1: str = r"model\CNN_model.h5"


project_config = ProjectConfig()
