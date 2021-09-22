import csv
import datetime
import json
import os
from datetime import datetime
import numpy as np


class AnnotationInterpreter:
    def __init__(self):
        # Different annotation types
        self.me_aws = "ME_AWS"
        self.me_tsr = "ME_TSR"
        #self.me_car = "ME_Car"
        #self.gps = "GPS"
        # List containing all the different annotation types
        self.__annotation_types = [self.me_aws, self.me_tsr, self.me_car, self.gps]
        self.__folder_path = "./TFC_ISEL_DashCam/trips"  # Folder where the meta information is present
        self.__current_time = None  # Current frame timestamp
        self.__vid_general_info_dict = {}  # Dictionary of the .JSON file belonging to the video
        self.__csv_info_dict = {}  # Dictionary that keeps the meta information belonging to the video
        self.__trip_number = 0  # Video Trip number
        self.__curr_ts_info_dict = {}  # Dictionary that keeps the meta information of the current frame

    # Runs all the functions
    def initialize_video(self, vid_name):
        self.__vid_general_info_dict, self.__trip_number = self.__get_video_info_dict(vid_name, "")
        trip_files = self.__get_trip_csv_files(self.__trip_number)

        for annotation_type in self.__annotation_types:
            annotation_data = self.__get_meta_info(trip_files, self.__vid_general_info_dict, self.me_aws)
            if annotation_data is not None and len(annotation_data) != 0:
                self.__csv_info_dict[annotation_type] = annotation_data

    # Updates the current timestamp of the video each frame
    def update_timestamp(self, framerate):
        microseconds_to_update = round(1000000 / framerate)
        time_change = datetime.timedelta(microseconds=microseconds_to_update)
        updated_time = self.__current_time + time_change
        self.__event_tracker(updated_time)
        self.__current_time = updated_time

    # Obtains all the events that took place between the current frame and the next frame
    def __event_tracker(self, end_time):
        annotations_used = [self.me_aws]  # List containing the meta information types that are used
        for annotation in annotations_used:
            if annotation in self.__csv_info_dict.keys():
                curr_anon_dict = self.__csv_info_dict[annotation]
                timestamps = curr_anon_dict['ts']
                start_idx = -10
                end_idx = -10
                found_start = False
                for ts in timestamps:
                    date_time_ts = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")
                    if self.__current_time <= date_time_ts < end_time:
                        if not found_start:
                            found_start = True
                            start_idx = timestamps.index(ts)
                            end_idx = timestamps.index(ts) + 1
                        elif date_time_ts < end_time:
                            end_idx = timestamps.index(ts) + 1
                        else:
                            break

                if start_idx != -10 and end_idx != -10:
                    present_info_dict = {}
                    for item in curr_anon_dict.items():
                        print(item)
                        values = item[1]
                        if start_idx != end_idx:
                            present_info_dict[item[0]] = np.array(values)[start_idx:end_idx]
                        else:
                            present_info_dict[item[0]] = np.array(values)[start_idx]

                    self.__curr_ts_info_dict = present_info_dict.copy()

                else:
                    self.__curr_ts_info_dict = {}

    # Verifies if a JSON file contains a dictionary belonging to the video_name inserted and returns it together with the JSON file name
    def __video_exists_in_file(self, video_name, json_file_name):
        with open(self.__folder_path + "/" + str(json_file_name)) as file:
            data = json.load(file)

            for dictionary in data:
                uuid = dictionary['uuid']
                if uuid == video_name:
                    return dictionary, True, json_file_name

        return None, False, ""

    # Discovers the trip belonging to the video and returns the dictionary with information about the video and its trip number
    def __get_video_info_dict(self, video_name, trip_file_name, find_trip_name=True):
        video_dictionary = None
        video_exists_in_file = False

        json_files = [json_file for json_file in os.listdir(self.__folder_path) if json_file.endswith('.json')]

        if find_trip_name:
            for json_file in json_files:
                video_dictionary, video_exists_in_file, trip_file_name = self.__video_exists_in_file(video_name, json_file)

                if video_exists_in_file:
                    break
        else:
            video_dictionary, video_exists_in_file, trip_file_name = self.__video_exists_in_file(video_name, trip_file_name)

        if not video_exists_in_file:
            print("Video name not existent or not belonging to specified trip number")
            return
        self.__current_time = datetime.strptime(video_dictionary['vid_start'], "%Y-%m-%dT%H:%M:%S.%f%z")
        return video_dictionary, trip_file_name.split("_")[0]

    # Returns a list with the csv file names(if existent) belonging to a certain trip
    def __get_trip_csv_files(self, trip_name):
        cvs_files = [csv_file for csv_file in os.listdir(self.__folder_path) if csv_file.endswith('.csv')]
        csv_trip_files = [parsed_csv for parsed_csv in cvs_files if parsed_csv.split("_")[0] == trip_name]
        if len(csv_trip_files) == 0:
            print("Trip number does not exist")
            return []
        return csv_trip_files

    # Obtains the video meta information of any of the csv files
    def __get_meta_info(self, trip_files, video_dict, annotation_type):
        if annotation_type not in self.__annotation_types:
            print("Annotation type non existent")
            return

        me_aws_file = [csv_file for csv_file in trip_files if csv_file.split("-")[-1] == annotation_type + ".csv"]

        if len(me_aws_file) == 0:
            print("MobilEye Warnings file does not exist")
            return []

        elif len(me_aws_file) == 1:
            ts_start = datetime.strptime(video_dict['vid_start'], "%Y-%m-%dT%H:%M:%S.%f%z")
            ts_end = datetime.strptime(video_dict['vid_end'], "%Y-%m-%dT%H:%M:%S.%f%z")
            video_data = []
            with open(self.__folder_path + "/" + me_aws_file[0], newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    curr_timestamp = datetime.strptime(row['ts'], "%Y-%m-%dT%H:%M:%S.%f%z")
                    if curr_timestamp >= ts_start:
                        if curr_timestamp <= ts_end:
                            video_data.append(row)
                        else:
                            return format_video_data(video_data)

        return None


# Formats list of dictionaries to only a dictionary
def format_video_data(video_data):
    new_dict = {}
    for key in video_data[0].keys():
        key_values = []
        for dictionary in video_data:
            key_values.append(dictionary[key])

        new_dict[key] = key_values

    return new_dict


interpreter = AnnotationInterpreter()
interpreter.initialize_video("EVUfcV74AuhN6QkdRX9pmD")
# interpreter.update_timestamp(15)
