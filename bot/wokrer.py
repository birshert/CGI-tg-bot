import queue
import time
from threading import Thread
import asyncio

import cv2
import numpy as np


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
        self.model = None

    def predict(self, image):
        if self.model is None:
            time.sleep(10)
            return np.random.random(image.shape) * 255
        else:
            return self.model(image)

    def get_waiting_time(self):
        return round(self.task_queue.qsize() * 10 / 60, 2)

    def run(self) -> None:
        while True:
            image_path, chat_id = self.task_queue.get()
            image = cv2.imread(image_path)
            image = self.predict(image)

            cv2.imwrite(image_path, image)
            self.ready_queue.put((image_path, chat_id))

