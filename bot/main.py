import asyncio
import queue
import urllib

from aiogram import Bot
from aiogram import types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import os

from bot.config import BOT_TOKEN
from bot.wokrer import RecognizeThread, send_ready_images

bot = Bot(token=BOT_TOKEN)

dp = Dispatcher(bot)
task_queue = queue.Queue()


@dp.message_handler(commands=['start', 'help'])
async def some_handler(message: types.Message):
    start_message = "Привет! Это бот, который умеет преобразовывать фотографии лиц в мультяшные. \n" \
              "Просто пришли нам свою фотку, и мы сделаем из тебя диснеевскую принцессу)"

    await message.answer(start_message)


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message):
    document_id = message['photo'][-1].file_id

    file_info = await bot.get_file(document_id)
    filename, file_extension = os.path.splitext(file_info.file_path)
    image_name = os.path.join('photos', document_id + file_extension)
    urllib.request.urlretrieve(f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}', image_name)
    task_queue.put((image_name, message.chat.id))
    await message.answer("Ваше изображение обрабатывается, это займет примерно {} минут \n".format(recognize_thread.get_waiting_time()))


if __name__ == '__main__':
    recognize_thread = RecognizeThread(task_queue)
    recognize_thread.daemon = True
    recognize_thread.start()

    loop = asyncio.get_event_loop()
    loop.create_task(send_ready_images(recognize_thread.ready_queue, bot))

    executor.start_polling(dp, loop=loop)


