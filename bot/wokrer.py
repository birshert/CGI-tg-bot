import queue
import time
from threading import Thread
import asyncio

import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch

import dnnlib
import legacy
import predict
from metrics.metric_utils import get_feature_detector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_it_video(path):
    return '.MOV' in path or '.mp4' in path or '.avi' in path


async def send_ready_images(ready_queue, bot):
    while True:
        if ready_queue.qsize():
            image_path, chat_id = ready_queue.get()
            print('sending photo to {}'.format(chat_id))
            with open(image_path, 'rb') as image:
                await bot.send_photo(chat_id=chat_id, photo=image)
        else:
            await asyncio.sleep(1)


class RecognizeThread(Thread):
    def __init__(self, task_queue):
        Thread.__init__(self)
        self.task_queue = task_queue
        self.ready_queue = queue.Queue()
        self.batch_size = 4
        self.model = MTCNN(image_size=1024, select_largest=True, device=device)
        url1 = "https://github.com/birshert/CGI-tg-bot/releases/download/model/stylegan3-r-ffhq-1024x1024.pkl"
        url2 = "https://github.com/birshert/CGI-tg-bot/releases/download/cartoon-model/stylegan3-r-ffhq-1024x1024-cartoon.pkl"
        url3 = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl"
        with dnnlib.util.open_url(url1) as fp:
            self.G1 = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
        with dnnlib.util.open_url(url2) as fp:
            self.G2 = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
        with dnnlib.util.open_url(url3) as fp:
            self.vgg16 = get_feature_detector(url3).eval().to(device)

    def predict(self, images):
        res = []
        for image in images:
            image = Image.fromarray(image)
            image = (self.model(image) + 1) / 2 * 255
            print(image.max())
            image = image.permute(1, 2, 0).int().numpy()
            res.append(predict.predict_by_noise(self.G1, self.G2, self.vgg16, Image.fromarray(image), 228, 100))

        return res

    def get_waiting_time(self):
        return round(self.task_queue.qsize() * 10 / 60, 2)

    def predict_batch(self, batch, res):
        pred = self.predict(np.array(batch))
        res = np.stack([pred, res]) if res.size() else pred

    def predict_video(self, video_path, chat_id):
        cap = cv2.VideoCapture(video_path)

        save_path = 'res_' + video_path[:video_path.rfind('.')] + '.mp4'
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        batch = []
        res = np.array([])
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            batch.append(image)
            if len(batch) == self.batch_size:
                self.predict_batch(batch, res)
                batch = []

            k = cv2.waitKey(25)
            if k == 27:
                break
        if len(batch):
            self.predict_batch(batch, res)

        for image in res:
            out.write(image)

        out.release()
        cap.release()

    def run(self) -> None:
        while True:
            cnt = 0
            chats = []
            image_paths = []
            images = []
            while cnt < self.batch_size and self.task_queue.qsize():
                image_path, chat_id = self.task_queue.get()
                if is_it_video(image_path):
                    # self.predict_video(image_path, chat_id)
                    continue
                else:
                    image_paths.append(image_path)
                    chats.append(chat_id)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    cnt += 1

            pred = self.predict(images)
            for i, image in enumerate(pred):
                print(image.shape)
                print(image.min(), image.max())
                cv2.imwrite(image_paths[i], cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2BGR))
                self.ready_queue.put((image_paths[i], chats[i]))
