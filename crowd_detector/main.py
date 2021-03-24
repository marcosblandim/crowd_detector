import cv2
import os
import logging
import time
import yaml

from yolo_detection import detect_crowds


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

settings_filename = "settings.yaml"
with open(os.path.join(DIR_PATH, settings_filename)) as settings_file:
    settings = yaml.safe_load(settings_file)

folder_path = os.path.join(DIR_PATH, settings["foldername"])
file_path = os.path.join(folder_path, settings["filename"])

try:
    os.mkdir(folder_path)
except FileExistsError:
    pass

logging.basicConfig(
    level=settings["log_level"].upper(),
    format='[%(levelname)s] %(asctime)s : "%(message)s"',
)


def fetch_frame(source,max_tries=3):
    '''Tries to fetch one frame from the source max_tries times, and then returns it.
    '''
    for i in range(max_tries):
        logging.debug(f"entering try number {i+1}/{max_tries}")

        cap = cv2.VideoCapture(source)
        ret,frame = cap.read()
        
        if ret: break
    
    if not ret: logging.warning("could not fetch any frame; max tries reached")

    cap.release()
    return frame, ret


def main():
    logging.info("starting program")
    try:
        for frame_counter in range(settings["number_of_frames"]):

            frame, ret = fetch_frame(settings["frame_source"], settings["max_tries"])
            if not ret:
                max_tries = settings["max_tries"]
                logging.warning(f"failed at fetching frames, tried {max_tries} times")
                break

            _, marked_frame = detect_crowds(frame)

            cv2.imwrite(file_path.format(frame_counter), marked_frame)

            number_of_frames = settings["number_of_frames"]
            logging.info(f"frame {frame_counter + 1}/{number_of_frames} saved with success")

            time.sleep(settings["fetch_interval"])

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt exception, aborting")
        exit()

if __name__ == "__main__": 
    main()
    