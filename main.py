#!/usr/bin/env python3

from timestamp_parser import ImageDateTimeClassifier
from image_processing import VideoProcessor
from utils import load_config
import os

def main():
    folder_path = 'data/output_rec/rkbt/1'
    config_path = 'config.json'

    config = load_config(config_path)
    classifier = ImageDateTimeClassifier(
        model_path=config["model_path"],
        plate_bbox_relative=tuple(config["plate_bbox_relative"]),
        digit_positions=config["digit_positions"],
        digit_width=config["digit_width"],
        date_format=config["date_format"]
    )

    video_processor = VideoProcessor(folder_path, config_path, classifier)
    video_processor.process_videos()

    video_processor.output_total_seconds_by_hour()
    video_processor.output_timestamps_by_video()

if __name__ == "__main__":
    main()

