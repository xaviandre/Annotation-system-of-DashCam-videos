import cv2
from shapely.geometry import Polygon, LineString, Point
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

plot_car_id = 12


class IntersectionOverUnionTracker:
    def __init__(self):
        # Store the bounding box points from the cars
        self.bb_points = {}
        self.bb_areas = {}
        self.bb_distances = {}
        self.bb_intersections = {}
        # Keep the count of the IDs
        # each time a new car is detected, the count will increase by one
        self.id_count = 0
        self.id_count_check = 0

    def __add_car(self, vehicle, bbs_ids, check_only=False):
        car = Polygon(vehicle)
        car = car.buffer(0)

        for car_id, car_points in self.bb_points.items():
            existent_car = Polygon(car_points)
            existent_car = existent_car.buffer(0)
            intersection = car.intersection(existent_car)
            union = car.union(existent_car)
            iou = round(((intersection.area / union.area) * 100), 2)

            # If car exists already, update correspondent ID bounding box points
            if iou > 40.00:
                if check_only:
                    return car_id
                else:
                    self.bb_points[car_id] = vehicle
                    bbs_ids.append([vehicle, car_id])
                    return bbs_ids

        # If the car does not exist, assign new ID to the car
        if check_only:
            return self.id_count
        else:
            self.bb_points[self.id_count] = vehicle
            bbs_ids.append([vehicle, self.id_count])
            self.id_count += 1
            return bbs_ids

    def update(self, image, curr_frame_idx, detected_cars, video_lanes, line_thickness=3, plot_cars=True):
        if len(detected_cars) > 0:
            # Cars bbs and IDs
            car_bbs_ids = []

            # Get points of each detected car
            for curr_car in detected_cars:
                # Find out if the car already exists
                car_bbs_ids = self.__add_car(curr_car, car_bbs_ids)

            # Clean the dictionary by removing IDs not used anymore
            new_bb_points = {}
            for vertices, car_id in car_bbs_ids:
                points = self.bb_points[car_id]
                new_bb_points[car_id] = points

            # Update dictionary with non used IDs removed
            self.bb_points = new_bb_points.copy()

            if plot_cars:
                self.__plot_car_ids(image, line_thickness)

            self.__save_bb_features(curr_frame_idx, video_lanes)

    def __save_bb_features(self, curr_frame_idx, video_lanes):
        # Fill the bounding box sizes list of the non present cars with None or the bb area for existing cars
        if len(self.bb_areas) > 0:
            # Get the IDs of the new cars in the present frame
            new_car_ids = list(set(self.bb_points.keys()) - set(self.bb_areas.keys()))
            # Fill the lists with the bb area or None, when the car is no longer present in the frame
            for car_id in self.bb_areas.keys():
                car_area = None
                car_distance = None
                car_intersection = None
                if car_id in self.bb_points.keys():
                    car = self.bb_points[car_id]
                    car_bb = Polygon(car)
                    car_area = car_bb.area
                    car_distance = get_distance_in_frame(car, car_area, video_lanes)
                    car_intersection = get_intersection_value(car, video_lanes)
                self.bb_areas[car_id].append(car_area)
                self.bb_distances[car_id].append(car_distance)
                self.bb_intersections[car_id].append(car_intersection)

            # Add to the bb sizes dictionary the new cars present in the current frame
            for car_id in new_car_ids:
                car = self.bb_points[car_id]
                car_bb = Polygon(car)
                car_area = car_bb.area
                bb_size = [None] * (curr_frame_idx - 1)
                bb_size.append(car_area)
                self.bb_areas[car_id] = bb_size

                car_distance = get_distance_in_frame(car, car_area, video_lanes)
                bb_distance = [None] * (curr_frame_idx - 1)
                bb_distance.append(car_distance)
                self.bb_distances[car_id] = bb_distance

                car_intersection = get_intersection_value(car, video_lanes)
                bb_intersection = [None] * (curr_frame_idx - 1)
                bb_intersection.append(car_intersection)
                self.bb_intersections[car_id] = bb_intersection

        # In case there are still no car bb sizes added to the dictionary
        else:
            for car_id, bb in self.bb_points.items():
                car = self.bb_points[car_id]
                car_bb = Polygon(car)
                car_area = car_bb.area
                bb_size = [None] * (curr_frame_idx - 1)
                bb_size.append(car_area)
                self.bb_areas[car_id] = bb_size

                car_distance = get_distance_in_frame(car, car_area, video_lanes)
                bb_distance = [None] * (curr_frame_idx - 1)
                bb_distance.append(car_distance)
                self.bb_distances[car_id] = bb_distance

                car_intersection = get_intersection_value(car, video_lanes)
                bb_intersection = [None] * (curr_frame_idx - 1)
                bb_intersection.append(car_intersection)
                self.bb_intersections[car_id] = bb_intersection

    def get_bb_area_variance(self, save_dir):
        if plot_car_id in self.bb_areas.keys():
            bb_area_values = self.bb_areas[plot_car_id]
            bb_distance_values = self.bb_distances[plot_car_id]
            bb_intersection_values = self.bb_intersections[plot_car_id]

            i = 0
            x = list()
            y_area = list()
            y_distance = list()
            y_intersection = list()
            for bb_value in bb_area_values:
                if bb_value is not None:
                    x.append(i)
                    y_area.append(bb_value)
                    y_distance.append(bb_distance_values[i])
                    y_intersection.append(bb_intersection_values[i])
                i += 1

            plt.plot(x, y_area)
            plt.savefig(f"{save_dir}/Area car {plot_car_id}.png")
            plt.clf()

            plt.plot(x, y_distance)
            plt.savefig(f"{save_dir}/Distance car {plot_car_id}.png")
            plt.clf()

            plt.plot(x, y_intersection)
            plt.savefig(f"{save_dir}/Intersection car {plot_car_id}.png")
            plt.clf()

            n = 50  # the larger n is, the smoother curve will be
            b = [1.0 / n] * n

            yy_area = lfilter(b, 1, np.gradient(y_area))
            plt.plot(x, yy_area)
            plt.savefig(f"{save_dir}/Area derivative car {plot_car_id}.png")
            plt.clf()

            yy_distance = lfilter(b, 1, np.gradient(y_distance))
            plt.plot(x, yy_distance)
            plt.savefig(f"{save_dir}/Distance derivative car {plot_car_id}.png")
            plt.clf()

    def get_vehicle_label_points(self, vehicle, image, line_thickness=3):
        car_id = self.__add_car(vehicle, None, check_only=True)
        tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        label = "" + str(car_id) + "0"
        font_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]  # line/font thickness
        car_label_coords = (vehicle[0][0] + font_size[0] + 1, vehicle[0][1] - 2)
        car_label_rect_coords = [(vehicle[0][0] + font_size[0], vehicle[0][1]),
                                 (vehicle[0][0] + font_size[0] + 1, vehicle[0][1] - font_size[1] - 3)]

        return car_label_coords, car_label_rect_coords

    def __plot_car_ids(self, image, line_thickness):
        if len(self.bb_points) > 0:
            tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
            tf = max(tl - 1, 1)  # font thickness

            for car_id, car_points in self.bb_points.items():
                label_car_id = "" + str(car_id)
                font_size = cv2.getTextSize(label_car_id, 0, fontScale=tl / 3, thickness=tf)[0]  # line/font thickness

                p_text = (car_points[0][0], car_points[0][1] - 2)
                p1 = (car_points[0][0], car_points[0][1])
                p2 = (car_points[0][0] + font_size[0], car_points[0][1] - font_size[1] - 3)

                cv2.rectangle(image, p1, p2, [74, 207, 237], -1, cv2.LINE_AA)
                cv2.putText(image, label_car_id, p_text, 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


def get_intersection_value(car, video_lanes):
    lane_vertices = np.array([[video_lanes[-1][0][0], video_lanes[-1][0][1]], [video_lanes[-1][0][2], video_lanes[-1][0][3]],
                              [video_lanes[-1][1][2], video_lanes[-1][1][3]], [video_lanes[-1][1][0], video_lanes[-1][1][1]]])
    lane_bb = Polygon(np.reshape(lane_vertices, (4, 2))).buffer(0)
    car_bb = Polygon(car).buffer(0)
    percentage = 0
    if lane_bb.intersects(car_bb):
        intersection = car_bb.intersection(lane_bb).area
        percentage = int((intersection / car_bb.area) * 100)

    return percentage


def get_distance_in_frame(car, car_area, video_lanes):
    cx_car = (car[0][0] + car[1][0]) // 2
    cy_car = (car[0][1] + car[1][1]) // 2
    c_car = Point(cx_car, cy_car)
    rect_lane = video_lanes[-1]
    lane_points = [[(rect_lane[0][0] + rect_lane[1][0]) // 2, (rect_lane[0][1] + rect_lane[1][1]) // 2],
                   [(rect_lane[0][2] + rect_lane[1][2]) // 2, (rect_lane[0][3] + rect_lane[1][3]) // 2]]
    lane = LineString(lane_points)
    return (c_car.distance(lane) * c_car.distance(lane)) / car_area
