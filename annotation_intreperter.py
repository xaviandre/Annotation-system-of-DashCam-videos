import os, json

class AnnotationIntreperter:

    def __init__(self):
        self.annotation_type = ""
        self.ano_types = ["ME_AWS", "ME_Car", "ME_TSR", "GPS", "DrivingEvents_Map"]
        self.folder_path = "./TFC_ISEL_Dashcam/trips"

    def __video_exists_in_file(self, video_name, json_file_name):
        with open(self.folder_path + "/" + str(json_file_name)) as file:
            data = json.load(file)

            for dictionary in data:
                uuid = dictionary['uuid']
                if uuid == video_name:
                    return dictionary, True, json_file_name

        return None, False, ""

    # Discovers the trip belonging to the video and returns the dictionary with information about the video and its trip number
    def get_video_info_dict(self, video_name, trip_file_name, find_trip_name=True):
        video_dictionary = None
        video_exists_in_file = False

        json_files = [json_file for json_file in os.listdir(self.folder_path) if json_file.endswith('.json')]
        print("Ficheiros JSON: ", json_files)

        if find_trip_name:
            for json in json_files:
                video_dictionary, video_exists_in_file, trip_file_name = self.__video_exists_in_file(video_name, json)

                if video_exists_in_file:
                    break
        else:
            video_dictionary, video_exists_in_file, trip_file_name = self.__video_exists_in_file(video_name, trip_file_name)

        if not video_exists_in_file:
            print("Nome do vídeo inválido ou não pertencente à viagem definida")
            return

        return video_dictionary, trip_file_name.split("_")[0]

intreperter = AnnotationIntreperter()
ayy, ayy1 = intreperter.get_video_info_dict("EVUfcV74AuhN6QkdRX9pmD", "")
print(ayy1)