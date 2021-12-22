import os
import queue
import urllib

from aiogram import Bot
from aiogram import types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from config import BOT_TOKEN


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
    file_id = message['photo'][-1].file_unique_id

    file_info = await bot.get_file(document_id)
    filename, file_extension = os.path.splitext(file_info.file_path)
    image_name = os.path.join('photos', file_id + file_extension)
    urllib.request.urlretrieve(f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}', image_name)
    task_queue.put((image_name, message.chat.id))

    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(InlineKeyboardButton(":thumbs_up:", callback_data="yes"),
               InlineKeyboardButton(":thumbs_down:", callback_data=f"no_{image_name}"))

    with open(image_name, 'rb') as image:
        await bot.send_photo(chat_id=message['from']['id'], photo=image, reply_markup=markup)


@dp.callback_query_handler(lambda x: x.data.startswith('no'))
async def foo(call: types.CallbackQuery):
    os.remove(call.data[3:])


if __name__ == '__main__':
    executor.start_polling(dp)
