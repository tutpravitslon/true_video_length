import os
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import cv2

class VideoProcessor:
    def __init__(self, folder_path, config_path, classifier):
        self.folder_path = folder_path
        self.classifier = classifier
        self.total_seconds_by_hour = defaultdict(int)
        self.timestamps_by_video = defaultdict(list)

        self.logger = logging.getLogger('video_processor')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)

    def capture_first_and_last_frames(self, video_path):
        """Capture and return the first and last frames of a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Error: cannot open video {video_path}")
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first_frame = cap.read()
        if not ret:
            self.logger.error(f"Error: cannot read first frame of {video_path}")
            cap.release()
            return None, None

        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        if not ret:
            self.logger.error(f"Error: cannot read last frame of {video_path}")
            cap.release()
            return None, None

        cap.release()
        return first_frame, last_frame

    def process_video_file(self, video_path):
        """Process a single video file to calculate intervals by hour and store timestamps/durations."""
        first_frame, last_frame = self.capture_first_and_last_frames(video_path)
        if first_frame is None or last_frame is None:
            return

        first_timestamp = self.classifier.parse_timestamp(first_frame)
        last_timestamp = self.classifier.parse_timestamp(last_frame)
        duration_seconds = last_timestamp - first_timestamp

        self.timestamps_by_video[os.path.basename(video_path)].append({
            'first_timestamp': first_timestamp,
            'last_timestamp': last_timestamp,
            'duration_seconds': duration_seconds
        })

        current_time = first_timestamp
        while current_time < last_timestamp:
            current_datetime = datetime.fromtimestamp(current_time)
            current_hour = current_datetime.hour
            next_hour_datetime = current_datetime.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            next_hour_timestamp = next_hour_datetime.timestamp()

            if next_hour_timestamp > last_timestamp:
                interval = last_timestamp - current_time
            else:
                interval = next_hour_timestamp - current_time

            self.total_seconds_by_hour[current_hour] += interval
            current_time = next_hour_timestamp

    def process_videos(self):
        """Process all video files in a folder to calculate total seconds by hour and store timestamps/durations."""
        if not os.path.isdir(self.folder_path):
            raise ValueError("Error: folder does not exist.")
        
        video_files = [f for f in os.listdir(self.folder_path) if f.endswith('.mp4')]

        for video_file in video_files:
            video_path = os.path.join(self.folder_path, video_file)
            self.process_video_file(video_path)

    def output_timestamps_by_video(self):
        total_duration_all = 0 
        for video_file, timestamps_list in sorted(self.timestamps_by_video.items()):
            self.logger.info(f"Timestamps for '{video_file}':")

            total_duration = 0
            for idx, timestamps_info in enumerate(timestamps_list):
                first_timestamp = timestamps_info['first_timestamp']
                last_timestamp = timestamps_info['last_timestamp']
                duration_seconds = timestamps_info['duration_seconds']

                timestamp_dt_start = datetime.fromtimestamp(first_timestamp)
                timestamp_dt_end = datetime.fromtimestamp(last_timestamp)

                self.logger.info(f"Interval {idx + 1}:")
                self.logger.info(f"  - Start: {timestamp_dt_start.strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"  - End: {timestamp_dt_end.strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"  - Duration (seconds): {duration_seconds}")

                total_duration += duration_seconds
                total_duration_all += duration_seconds

            self.logger.info(f"Total Duration for '{video_file}': {total_duration} seconds")
            self.logger.info("") 
            
        self.logger.info(f"Total Duration for '{video_file}': {total_duration_all} seconds")

    def output_total_seconds_by_hour(self):
        for hour, seconds in sorted(self.total_seconds_by_hour.items()):
            self.logger.info(f"Total time in hour {hour:02d}: {seconds} seconds")
