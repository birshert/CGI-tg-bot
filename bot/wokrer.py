import asyncio
import queue
import sys
from threading import Thread

import cv2
import dnnlib
import legacy
import numpy as np
import predict
import torch
from metrics.metric_utils import get_feature_detector
from PIL import Image


sys.path.append('face-alignment')

from face_aligner import align_image
from face_aligner import get_detector


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_it_video(path):
    return '.MOV' in path or '.mp4' in path or '.avi' in path


async def send_ready_images(ready_queue, bot):
    while True:
        if ready_queue.qsize():
            path, chat_id = ready_queue.get()
            if is_it_video(path):
                print('sending video to {}'.format(chat_id))
                with open(path, 'rb') as f:
                    await bot.send_video(chat_id=chat_id, video=f)
            else:
                print('sending photo to {}'.format(chat_id))
                with open(path, 'rb') as f:
                    await bot.send_photo(chat_id=chat_id, photo=f)
        else:
            await asyncio.sleep(0.1)


class RecognizeThread(Thread):

    def __init__(self, task_queue):
        Thread.__init__(self)
        self.task_queue = task_queue
        self.ready_queue = queue.Queue()
        self.batch_size = 4

        self.landmarks_detector = get_detector()

        url1 = "Stylegan3/models/stylegan3-r-ffhq-1024x1024.pkl"
        url2 = "Stylegan3/models/stylegan3-r-ffhq-1024x1024-cartoon.pkl"
        url3 = "Stylegan3/models/vgg16.pkl"

        print('LOADING MODELS')

        with dnnlib.util.open_url(url1) as fp:
            self.G1 = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
        with dnnlib.util.open_url(url2) as fp:
            self.G2 = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
        self.vgg16 = get_feature_detector(url3).eval().to(device)

        print('MODELS LOADED')

    def align_face_on_image(self, image_path):
        for face in align_image(self.landmarks_detector, image_path):
            yield face

    def predict(self, image_paths):
        for image_path in image_paths:
            for face in self.align_face_on_image(image_path):
                yield image_path, predict.predict_by_noise(self.G1, self.G2, self.vgg16, face, 0, 10)

    def get_waiting_time(self):
        return round(self.task_queue.qsize() * 10 / 60, 2)

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        save_path = video_path[:video_path.rfind('.')] + '_anim.mp4'
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        freq = fps // 10
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(save_path, fourcc, 10, (1024, 1024))

        batch = []
        cnt = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            cnt += 1
            if cnt % freq != 0:
                continue

            batch.append(Image.fromarray(image))
            if len(batch) == self.batch_size:
                for image in self.predict(batch):
                    print(np.array(image).shape)
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    out.write(image)

                batch = []

        if len(batch):
            for image in self.predict(np.array(batch)):
                out.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        out.release()
        cap.release()
        print('video released')
        return save_path

    def run(self) -> None:
        image_path_counter = {}

        while True:
            cnt = 0

            chats = []
            image_paths = []

            while cnt < self.batch_size and self.task_queue.qsize():
                image_path, chat_id = self.task_queue.get()

                if is_it_video(image_path):
                    path = self.predict_video(image_path)
                    self.ready_queue.put((path, chat_id))
                    continue
                else:
                    image_paths.append(image_path)
                    chats.append(chat_id)

                    cnt += 1

            for i, (image_path, image) in enumerate(self.predict(image_paths)):
                if image_path in image_path_counter:
                    image_path_counter[image_path] += 1
                else:
                    image_path_counter[image_path] = 0

                image_path = image_path[:image_path.rfind('.')] + f'{image_path_counter[image_path]:03d}.' + '.png'
                image.save(image_path)
                self.ready_queue.put((image_path, chats[i]))
