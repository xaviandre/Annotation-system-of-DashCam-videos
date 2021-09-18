import os, json, csv
from datetime import datetime
import datetime as dtime

class AnnotationIntreperter:

    def __init__(self):
        self.me_aws = "ME_AWS"
        self.me_tsr = "ME_TSR"
        self.me_car = "ME_Car"
        self.gps = "GPS"
        self.drivingevents_map = "DrivingEvents_Map"
        self.__anot_types = [self.me_aws, self.me_tsr, self.me_car, self.gps, self.drivingevents_map]
        self.__folder_path = "./TFC_ISEL_Dashcam/trips"
        self.__current_time = None
        self.__video_dict = {}
        self.__info_dict = {}
        self.__trip_number = 0

    def initialize_video(self, vid_name):
        self.__video_dict, self.__trip_number = self.__get_video_info_dict(vid_name, "")
        trip_files = self.__get_trip_csv_files(self.__trip_number)
        for anot in self.__anot_types:
            anot_data = self.get_meta_info(trip_files, self.__video_dict, self.me_aws)
            if anot_data != None or len(anot_data) == 0:
                self.__info_dict[anot] = anot_data

    # Updates the current timestamp of the video each frame
    def update_timestamp(self, framerate):
        microseconds_to_update = round(1000000/framerate)
        time_change = dtime.timedelta(microseconds=microseconds_to_update)
        self.__current_time = self.__current_time + time_change

    def __event_tracker(self):
        #TODO Detetar quando existem eventos ocurridos entre a frame atual e a anterior, talvez definir um step
        hello = "world"

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
            for json in json_files:
                video_dictionary, video_exists_in_file, trip_file_name = self.__video_exists_in_file(video_name, json)

                if video_exists_in_file:
                    break
        else:
            video_dictionary, video_exists_in_file, trip_file_name = self.__video_exists_in_file(video_name, trip_file_name)

        if not video_exists_in_file:
            print("Video name not existent or not belonging to specified trip number")
            return
        self.__current_time = datetime.strptime(video_dictionary['vid_start'], "%Y-%m-%dT%H:%M:%S.%f%z")
        return video_dictionary, trip_file_name.split("_")[0]

    def __get_trip_csv_files(self, trip_name):
        cvs_files = [csv_file for csv_file in os.listdir(self.__folder_path) if csv_file.endswith('.csv')]
        csv_trip_files = [parsed_csv for parsed_csv in cvs_files if parsed_csv.split("_")[0] == trip_name]
        if len(csv_trip_files) == 0:
            print("Trip number does not exist")
            return []
        return csv_trip_files

    # Obtains the video meta information of any of the csv files
    def get_meta_info(self, trip_files, video_dict, anotation_type):
        if anotation_type not in self.__anot_types:
            print("Annotation type non existent")
            return

        me_aws_file = [csv_file for csv_file in trip_files if csv_file.split("-")[-1] == anotation_type + ".csv"]

        if len(me_aws_file) == 0:
            print("Mobileye Warnings file does not exist")
            return []

        elif len(me_aws_file) == 1:
            ts_start = datetime.strptime(video_dict['vid_start'], "%Y-%m-%dT%H:%M:%S.%f%z")
            ts_end = datetime.strptime(video_dict['vid_end'], "%Y-%m-%dT%H:%M:%S.%f%z")
            video_data = []
            with open(self.__folder_path + "/" + me_aws_file[0], newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    curr_timestamp = datetime.strptime(row['ts'], "%Y-%m-%dT%H:%M:%S.%f%z")
                    if curr_timestamp >= ts_start:
                        if curr_timestamp <= ts_end:
                            video_data.append(row)
                        else:
                            return video_data

intreperter = AnnotationIntreperter()
intreperter.initialize_video("EVUfcV74AuhN6QkdRX9pmD")