import argparse
import json
import math
import os
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import cv2
from PIL import Image
sat_orin_path = "/export/cc/SA-OCC/data/sat/"



EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}

gps_ranges = {
    "boston-seapot": {
        "min": (-71.05851008588972, 42.33797303172644),
        "max": (-71.02355110151471, 42.35621253138901)
    },
    "singapore-hollandvillage": {
        "min": (103.7830192508019, 1.3045290611255334),
        "max": (103.80699190705191, 1.328495917110021)
    },
    "singapore-onenorth": {
        "min": (103.78398168303634, 1.2863845588122684),
        "max": (103.79696801116135, 1.3103515735653453)
    },
    "singapore-queenstown": {
        "min": (103.7687112761037, 1.2850861265683637),
        "max": (103.79268393235371, 1.3094289254199807)
    }
}

sat_out_path = "/export/cc/SA-OCC/data/nuscenes/sat/"

# 遍历字典中的键
for key in gps_ranges.keys():
    # 构造图像文件名
    image_filename = sat_orin_path + f"{key}.png"
    
    # 尝试打开图像文件
    try:
        with Image.open(image_filename) as img:
            # 获取图像的高度和宽度
            height = img.height
            width = img.width
            
            # 将宽度和高度信息添加到字典中
            gps_ranges[key]["W"] = width
            gps_ranges[key]["H"] = height

            # 输出图像的高度和宽度
            print(f"{image_filename}: Height = {height}, Width = {width}")
    except FileNotFoundError:
        print(f"File {image_filename} not found.")
    except Exception as e:
        print(f"An error occurred while opening {image_filename}: {e}")

# 打印更新后的字典
print("Updated gps_ranges dictionary:")
for key, value in gps_ranges.items():
    print(f"{key}: {value}")

# 判断位置属于哪个GPS范围，并返回键
def find_key_for_location(gps_ranges, location):
    lon, lat = location
    for key, range_info in gps_ranges.items():
        if (range_info["min"][0] <= lon and lon <= range_info["max"][0] and
                range_info["min"][1] <= lat and lat <= range_info["max"][1]):
            x = (lon - range_info["min"][0]) / (range_info["max"][0] - range_info["min"][0]) * range_info["W"]
            y = (1 - (lat - range_info["min"][1]) / (range_info["max"][1] - range_info["min"][1])) * range_info["H"]
            center_point  = (x,y)

            return key, center_point

    print("ERROR", location)
    return None

def get_poses(nusc: NuScenes, scene_token: str) -> List[dict]:
    """
    Return all ego poses for the current scene.
    :param nusc: The NuScenes instance to load the ego poses from.
    :param scene_token: The token of the scene.
    :return: A list of the ego pose dicts.
    """
    pose_list = []
    scene_rec = nusc.get('scene', scene_token)
    sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])

    
    ego_pose = nusc.get('ego_pose', sd_rec['token'])
    pose_list.append(ego_pose)

    while sd_rec['next'] != '':
        sd_rec = nusc.get('sample_data', sd_rec['next'])
        filename = sd_rec['filename']
        # print(filename, filename.split('/')[0])
        if filename.split('/')[0] == 'sweeps':
            continue
        ego_pose = nusc.get('ego_pose', sd_rec['token'])
        pose_list.append(ego_pose)

    return pose_list


def get_coordinate(ref_lat: float, ref_lon: float, bearing: float, dist: float) -> Tuple[float, float]:
    """
    Using a reference coordinate, extract the coordinates of another point in space given its distance and bearing
    to the reference coordinate. For reference, please see: https://www.movable-type.co.uk/scripts/latlong.html.
    :param ref_lat: Latitude of the reference coordinate in degrees, ie: 42.3368.
    :param ref_lon: Longitude of the reference coordinate in degrees, ie: 71.0578.
    :param bearing: The clockwise angle in radians between target point, reference point and the axis pointing north.
    :param dist: The distance in meters from the reference point to the target point.
    :return: A tuple of lat and lon.
    """
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    angular_distance = dist / EARTH_RADIUS_METERS
    
    target_lat = math.asin(
        math.sin(lat) * math.cos(angular_distance) + 
        math.cos(lat) * math.sin(angular_distance) * math.cos(bearing)
    )
    target_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
        math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat)
    )
    return math.degrees(target_lat), math.degrees(target_lon)


def derive_latlon(location: str, poses: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    For each pose value, extract its respective lat/lon coordinate and timestamp.
    
    This makes the following two assumptions in order to work:
        1. The reference coordinate for each map is in the south-western corner.
        2. The origin of the global poses is also in the south-western corner (and identical to 1).

    :param location: The name of the map the poses correspond to, ie: 'boston-seaport'.
    :param poses: All nuScenes egopose dictionaries of a scene.
    :return: A list of dicts (lat/lon coordinates and timestamps) for each pose.
    """
    assert location in REFERENCE_COORDINATES.keys(), \
        f'Error: The given location: {location}, has no available reference.'
    
    coordinates = []
    reference_lat, reference_lon = REFERENCE_COORDINATES[location]
    for p in poses:
        # print(p)
        ts = p['timestamp']
        x, y = p['translation'][:2]
        bearing = math.atan(x / y)
        distance = math.sqrt(x**2 + y**2)
        lat, lon = get_coordinate(reference_lat, reference_lon, bearing, distance)

        coordinates.append({'timestamp': ts, 'latitude': lat, 'longitude': lon, 'rot': p['rotation'], 'token': p['token']})
    return coordinates

import math
def get_rectangle_corners(center, size, rotation):
    """Calculate the corners of a rectangle given its center, size, and rotation."""
    W, H = size
    rotation_matrix = Quaternion(rotation).rotation_matrix[:2, :2].T
    # yaw, pitch, roll = Quaternion(rotation).yaw_pitch_roll
    # print(rotation, yaw/math.pi*180)
    
    # Define the half-size vectors
    half_x = np.array([W / 2, 0])
    half_y = np.array([0, H / 2])
    center = np.array([center[0], center[1]])
    # Rotate the half-size vectors
    rotated_half_x = np.dot(rotation_matrix, half_x)
    rotated_half_y = np.dot(rotation_matrix, half_y)
    
    # Calculate the corners
    corners = np.array([
        center - rotated_half_x - rotated_half_y,
        center + rotated_half_x - rotated_half_y,
        center + rotated_half_x + rotated_half_y,
        center - rotated_half_x + rotated_half_y
    ])
    # corners = np.array([
        # center - half_x - half_y,
    #     center + half_x - half_y,
    #     center + half_x + half_y,
    #     center - half_x + half_y
    # ])
    # corners = corners[:, :2]
    return corners

def crop_quadrilateral(img, corners):

    # corners = sort_points_clockwise_to_corners(corners)
    width = 400
    height = 400
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    corners = corners.astype(dst.dtype)
    # print(corners.shape)  # 应该输出 (4, 2)
    # print(corners.dtype)  # 应该输出 float32
    # print(dst.shape)  # 应该输出 (4, 2)
    # print(dst.dtype)  # 应该输出 float32
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(corners, dst)
    
    # 执行透视变换
    warped = cv2.warpPerspective(img, M, (width, height))
    
    return warped


def main(dataroot: str, version: str, output_prefix: str, output_format: str = 'kml', range=None) -> None:
    """
    Extract the latlon coordinates for each available pose and write the results to a file.
    The file is organized by location and scene_name.
    :param dataroot: Path of the nuScenes dataset.
    :param version: NuScenes version.
    :param output_format: The output file format, kml or json.
    :param output_prefix: Where to save the output file (without the file extension).
    """
    # Init nuScenes.
    nusc = NuScenes(dataroot=dataroot, version=version, verbose=False)
    count = 0
    count0 = 0
    size = (400, 400)

    imgs = {}
    for key in gps_ranges.keys():
        print(key)
        imgs[key] = cv2.imread(sat_orin_path + key + ".png")
            

    print(f'Extracting coordinates...')
    for scene in tqdm(nusc.scene):
        # Retrieve nuScenes poses.
        scene_name = scene['name']
        scene_token = scene['token']
        location = nusc.get('log', scene['log_token'])['location']  # Needed to extract the reference coordinate.
        poses = get_poses(nusc, scene_token)  # For each pose, we will extract the corresponding coordinate.
        # print('111111111111111111111:',location)

        # Compute and store coordinates.
        coordinates = derive_latlon(location, poses)
        # print(coordinates)
        print(len(coordinates))

        for dict in coordinates:
            count0 = count0 + 1
            print(count0)
            loc = (dict["longitude"], dict["latitude"])
            rot = dict["rot"]
            token = dict["token"]
            sample_data = nusc.get('sample_data', token)
            sample_token = sample_data['sample_token']
            sample = nusc.get('sample', sample_token)

            # print(token, sample['token'])
            key, center_point = find_key_for_location(range, loc)

            corners = get_rectangle_corners(center_point, size, rot)
            cropped_image = crop_quadrilateral(imgs[key], corners)

            out_path = sat_out_path + scene['name'] +'/' + sample['token']
            print(out_path)

            if not os.path.exists(out_path):
                os.makedirs(out_path)
                      
            # Save or display the cropped image
            cropped_image2 = Image.fromarray(cropped_image)
            cropped_image2.save(out_path + '/sat.png')
            # cropped_image2.save(sat_out_path + "/" + scene_name + str(count0) + '.png')

            # print(key, center_point)
            if key is None:
                count = count + 1
            

    print(count, count0, count/count0)



if __name__ == "__main__":

    # get_map_max_lon_lat(gps_ranges)
    parser = argparse.ArgumentParser(description='Export ego pose coordinates from a scene to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/export/cc/SA-OCC/data/nuscenes', help="Path where nuScenes is saved.")
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version.')
    parser.add_argument('--output_prefix', type=str, default='latlon',
                        help='Output file path without file extension.')
    parser.add_argument('--output_format', type=str, default='kml', help='Output format (kml or json).')
    args = parser.parse_args()

    main(args.dataroot, args.version, args.output_prefix, args.output_format, gps_ranges)

