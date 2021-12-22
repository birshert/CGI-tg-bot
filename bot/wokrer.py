import asyncio
import queue
import sys


sys.path.append('face-alignment')
sys.path.append('Stylegan3')
from threading import Thread

import dnnlib
import legacy
import predict
import torch
from dataclasses import dataclass
from metrics.metric_utils import get_feature_detector
from PIL import Image
from blend_models import blend

from face_aligner import align_image
from face_aligner import get_detector
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


@dataclass
class Task:
    chat_id: str
    image_path: str
    steps: int
    cnt_faces: int
    crop: bool


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


async def send_ready_images(ready_queue, bot):
    while True:
        if ready_queue.qsize():
            path, task = ready_queue.get()
            if path is None:
                await bot.send_message(
                    chat_id=task.chat_id,
                    text='Не нашли ни одного лица, попробуйте другую фотографию'
                )
                continue

            print('sending photo to {}'.format(task.chat_id))

            markup = InlineKeyboardMarkup()

            if task.crop:
                markup.row_width = 1
                markup.add(
                    InlineKeyboardButton('Обработать', callback_data='yes0_' + path),
                )
            else:
                markup.row_width = 2
                markup.add(
                    InlineKeyboardButton('\U0001F44D', callback_data='yes1_' + task.image_path),
                    InlineKeyboardButton('\U0001F44E', callback_data='no1_' + task.image_path)
                )

            with open(path, 'rb') as f:
                await bot.send_photo(
                    chat_id=task.chat_id,
                    photo=f,
                    reply_markup=markup
                )
        else:
            await asyncio.sleep(0.1)


class RecognizeThread(Thread):

    def __init__(self, task_queue):
        Thread.__init__(self)
        self.task_queue = task_queue
        self.ready_queue = queue.Queue()

        print('LOADING MODELS')

        self.landmarks_detector = get_detector()

        with dnnlib.util.open_url('Stylegan3/models/stylegan3-r-ffhq-1024x1024.pkl') as fp:
            self.G1 = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device).eval()

        self.G2 = blend(
            'Stylegan3/models/stylegan3-r-ffhq-1024x1024.pkl',
            'Stylegan3/models/stylegan3-r-ffhq-1024x1024.pkl'
        ).requires_grad_(False).to(device).eval()

        self.vgg16 = get_feature_detector('Stylegan3/models/vgg16.pkl').eval().to(device).eval()

        print('MODELS LOADED')

    def align_face_on_image(self, image_path: str):
        return list(align_image(self.landmarks_detector, image_path))

    def crop_faces(self, task: Task):
        image_path = task.image_path
        faces = self.align_face_on_image(image_path)
        if faces:
            for face in faces:
                save_path = image_path[:image_path.rfind('.')] + f'{task.cnt_faces:03d}' + '.png'
                task.cnt_faces += 1
                face.save(save_path)

                self.ready_queue.put((save_path, task))
        else:
            self.ready_queue.put((None, task))

    def predict(self, task: Task):
        image_path = task.image_path
        save_path = image_path[:image_path.rfind('.')] + '_g' + '.png'

        image = predict.predict_by_noise(self.G1, self.G2, self.vgg16, Image.open(image_path), 0, task.steps)
        image.save(save_path)

        self.ready_queue.put((save_path, task))
        
    def process_task(self, task: Task):
        if task.crop:
            self.crop_faces(task)
        else:
            self.predict(task)

    def get_waiting_time(self):
        return round(self.task_queue.qsize() * 10 / 60, 2)

    def run(self) -> None:
        while True:
            image_path, chat_id, steps, crop = self.task_queue.get()
            self.process_task(Task(chat_id=chat_id, image_path=image_path, steps=steps, cnt_faces=0, crop=crop))
