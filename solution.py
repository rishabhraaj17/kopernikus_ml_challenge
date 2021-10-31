import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from imaging_interview import compare_frames_change_detection, preprocess_image_change_detection

IMAGE_ROOT = 'ml-challenge/c23'

LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-30.30s   %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def plot_single_image(img: np.ndarray, c_map: Optional[str] = None) -> None:
    plt.imshow(img, cmap=c_map)
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_path', '-i', type=str, help='path to images to be filtered',
                        default='ml-challenge/c23')
    parser.add_argument('--output_path', '-o', type=str, help='path to save filtered images',
                        default='ml-challenge/filtered')
    parser.add_argument('--min_contour_area', '-ca', type=int, help='minimum contour area to consider', default=500)
    parser.add_argument('--border_mask', '-mask', type=tuple,
                        help='border mask values to mask out border of images', default=(0, 30, 0, 0))
    parser.add_argument('--blur_radius_list', '-blur_radius', type=tuple,
                        help='list of sigmas to apply gaussian blur', default=(21,))
    parser.add_argument('--duplicate_threshold', '-thresh', type=int,
                        help='threshold above which the image is not a duplicate', default=550)
    parser.add_argument('--contour_count', type=int,
                        help='restrict the duplicates by numbers of contours detected', default=6300)

    parsed_args = parser.parse_args()
    return parsed_args


class DuplicateRemover(object):
    def __init__(self, root_path: str, min_contour_area: int, black_mask: Tuple[int, int, int, int],
                 gaussian_blur_radius_list: Optional[List[int]], duplicate_threshold: int, dump_path: str,
                 contour_count: int = 1e9):
        super(DuplicateRemover, self).__init__()
        self.root_path = root_path
        self.dump_path = dump_path

        self.min_contour_area = min_contour_area
        self.black_mask = black_mask
        self.gaussian_blur_radius_list = gaussian_blur_radius_list
        self.duplicate_threshold = duplicate_threshold
        self.contour_count = contour_count

    @staticmethod
    def read_image(img_path: str) -> np.ndarray:
        return cv2.imread(img_path)

    def compare_images(self, img_path_0: str, img_path_1: str) -> Tuple[float, List[np.ndarray], float]:
        im1 = self.read_image(os.path.join(self.root_path, img_path_0))
        im2 = self.read_image(os.path.join(self.root_path, img_path_1))

        im1_gray = preprocess_image_change_detection(
            im1, gaussian_blur_radius_list=self.gaussian_blur_radius_list, black_mask=self.black_mask)
        im2_gray = preprocess_image_change_detection(
            im2, gaussian_blur_radius_list=self.gaussian_blur_radius_list, black_mask=self.black_mask)

        score, res_counts, thresh = compare_frames_change_detection(
            im1_gray, im2_gray, min_contour_area=self.min_contour_area)
        return score, res_counts, thresh

    def compare_all_images(self, dump_as_json: bool = True) -> Dict[str, Dict[str, Union[bool, str]]]:
        Path(self.dump_path).mkdir(parents=True, exist_ok=True)

        all_images_path = [img_name for img_name in sorted(os.listdir(self.root_path))]

        filtering_result = {
            f'{all_images_path[0]}': {
                'is_kept': True,
                'eliminated_by': None,
            }
        }
        shutil.copyfile(os.path.join(self.root_path, all_images_path[0]),
                        os.path.join(self.dump_path, all_images_path[0]))

        logger.info('Beginning Filtering: ')
        previous_image = all_images_path[0]
        for img_path in tqdm(range(1, len(all_images_path))):
            current_image = all_images_path[img_path]

            score, contours_count, _ = self.compare_images(previous_image, current_image)
            contours_count = np.concatenate(contours_count).shape[0] if len(contours_count) != 0 else 0

            is_kept = False
            eliminated_by = previous_image
            if score > self.duplicate_threshold and contours_count < self.contour_count:
                is_kept = True
                eliminated_by = None

                previous_image = current_image
                shutil.copyfile(os.path.join(self.root_path, current_image),
                                os.path.join(self.dump_path, current_image))

            filtering_result[f'{current_image}'] = {
                'is_kept': is_kept,
                'eliminated_by': eliminated_by
            }

        number_of_duplicates_found = len([k for k, v in filtering_result.items() if not v['is_kept']])
        if dump_as_json:
            file_dump_path = os.path.join(self.dump_path, 'results.json')

            with open(file_dump_path, 'w+') as f:
                json.dump({
                    'parameters': self.__dict__,
                    'number_of_duplicates_found': number_of_duplicates_found,
                    'results': filtering_result
                }, f, indent=4)
                logger.info(f"Filtering Results saved at {file_dump_path}")
                
        logger.info(f"Finished Filtering!\t{number_of_duplicates_found} duplicates found!")
        return filtering_result


def parameter_search(
        black_mask: Tuple[int, int, int, int],
        contour_area_list: List[int],
        list_of_blur_radius: List[Optional[List[int]]],
        duplicate_threshold_list: List[int],
        dump_root_path: str,
        contour_count_list: List[int] = 1e9) -> None:
    for blur_radius in list_of_blur_radius:
        for duplicate_threshold in duplicate_threshold_list:
            for contour_count in contour_count_list:
                for min_contour_area in contour_area_list:
                    folder_name = f'blur_radius_{blur_radius[0]}_dup_threshold_{duplicate_threshold}' \
                                  f'_contour_count_{contour_count}_min_contour_area_{min_contour_area}'
                    dump_path = f'{os.path.join(dump_root_path, folder_name)}'
                    Path(dump_path).mkdir(parents=True, exist_ok=True)

                    dup_remover = DuplicateRemover(
                        root_path=IMAGE_ROOT,
                        min_contour_area=min_contour_area,
                        black_mask=black_mask,
                        gaussian_blur_radius_list=blur_radius,
                        duplicate_threshold=duplicate_threshold,
                        dump_path=dump_path,
                        contour_count=contour_count
                    )
                    results = dup_remover.compare_all_images()


if __name__ == '__main__':
    args = parse_args()
    duplicate_remover = DuplicateRemover(
        root_path=args.input_path,
        min_contour_area=args.min_contour_area,
        black_mask=args.border_mask,
        gaussian_blur_radius_list=args.blur_radius_list,
        duplicate_threshold=args.duplicate_threshold,
        dump_path=args.output_path,
        contour_count=args.contour_count
    )
    out = duplicate_remover.compare_all_images()
