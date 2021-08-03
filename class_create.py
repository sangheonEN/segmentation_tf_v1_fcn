import json
import os, glob
import cv2
import numpy as np
from collections import OrderedDict

# json file에서 coordinate point data 불러오기
# 4points data를 가지고 대응되는 input image에 coordinate 그려지는거 확인 (좌표 검증)
# 좌표 검증되면 mask image에 해당 bounding box 그리고 내부 pixel 값을 해상 0 육지 1 ship 2로 변환.

json_file_path = './data/test_label/labels_test_origin.json'
image_path = './data/mask'
save_path = './data/save'

def json_file_load(json_path, image_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        data_num = len(json_data['features'])
        image_id = ''
        for i in range(data_num):
            json_bounds_imcoords = json_data['features'][i]['properties']['bounds_imcoords']
            json_image_id = json_data['features'][i]['properties']['image_id']

            x1min = float(json_bounds_imcoords.split(',')[0])
            x1min = round(x1min)

            y1min = float(json_bounds_imcoords.split(',')[1])
            y1min = round(y1min)

            x2min = float(json_bounds_imcoords.split(',')[2])
            x2min = round(x2min)

            y2min = float(json_bounds_imcoords.split(',')[3])
            y2min = round(y2min)

            x3min = float(json_bounds_imcoords.split(',')[4])
            x3min = round(x3min)

            y3min = float(json_bounds_imcoords.split(',')[5])
            y3min = round(y3min)

            x4min = float(json_bounds_imcoords.split(',')[6])
            x4min = round(x4min)

            y4min = float(json_bounds_imcoords.split(',')[7])
            y4min = round(y4min)

            img = cv2.imread(os.path.join(image_path, json_image_id), cv2.IMREAD_GRAYSCALE)

            if image_id != json_image_id:
                img = img / 255
                image_id = json_image_id
            else:
                img = img / 100

            point = np.array([[x1min, y1min], [x2min, y2min], [x3min, y3min],[x4min, y4min]])

            filled_poly = cv2.fillPoly(img, [point], (2))

            if image_id == json_image_id:
                filled_poly = filled_poly * 100

            cv2.imwrite(os.path.join(image_path, json_image_id), filled_poly)


if __name__ == "__main__":

    # json file 불러오기
    json_file_load(json_file_path, image_path)

