import urllib

from aiogram import Bot
from aiogram import types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import os

from bot.config import BOT_TOKEN

bot = Bot(token=BOT_TOKEN)

dp = Dispatcher(bot)


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

    with open(image_name, 'rb') as image:
        await bot.send_photo(chat_id=message.chat.id, photo=image)


if __name__ == '__main__':
    executor.start_polling(dp)
