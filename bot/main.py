import asyncio
import os
import queue
import urllib

from aiogram import Bot
from aiogram import types
from aiogram.dispatcher import Dispatcher
from aiogram.types import InlineKeyboardButton
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils import executor

from config import BOT_TOKEN
from worker import RecognizeThread
from worker import send_ready_images


bot = Bot(token=BOT_TOKEN)

dp = Dispatcher(bot)
task_queue = queue.Queue()

low_steps = 300
high_steps = 1000


@dp.message_handler(commands=['start', 'help'])
async def some_handler(message: types.Message):
    start_message = 'Привет! Это бот, который умеет преобразовывать фотографии лиц в мультяшные. \n' \
                    'Просто пришли нам свою фотку, и мы сделаем из тебя диснеевскую принцессу)'
    await message.answer(start_message)


@dp.message_handler(content_types=['photo'], state='*')
async def handle_docs_photo(message):
    document_id = message['photo'][-1].file_id
    file_id = message['photo'][-1].file_unique_id

    file_info = await bot.get_file(document_id)
    filename, file_extension = os.path.splitext(file_info.file_path)
    image_name = os.path.join('photos', file_id + file_extension)
    urllib.request.urlretrieve(f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}', image_name)
    task_queue.put((image_name, message.chat.id, low_steps, True))
    await message.answer('Ищем лица на вашем фото')


@dp.callback_query_handler(lambda x: x.data.startswith('yes'), state='*')
async def inline_yes1_answer_callback_handler(query: types.CallbackQuery):
    await bot.edit_message_reply_markup(query.message.chat.id, query.message.message_id, None)
    text = 'Отлично! Мы рады, что вам понравилось! Присылайте фотографии еще'
    img_path = query.data[5:]
    os.remove(img_path)
    await bot.send_message(query.from_user.id, text)


@dp.callback_query_handler(lambda x: x.data.startswith('no1_' + str(low_steps)), state='*')
async def inline_no1_answer_callback_handler(query: types.CallbackQuery):
    await bot.edit_message_reply_markup(query.message.chat.id, query.message.message_id, None)
    img_path = query.data[7:]
    text = 'Хотите ли вы, чтобы мы обработали фотографию чуть дольше, тем самым улучшив результат?'
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(
        InlineKeyboardButton('Да', callback_data='high' + img_path),
        InlineKeyboardButton('Нет', callback_data='no2_' + img_path)
    )
    await query.message.answer(text, reply_markup=markup)


@dp.callback_query_handler(lambda x: x.data.startswith('no1_' + str(high_steps)), state='*')
async def inline_no1_high_answer_callback_handler(query: types.CallbackQuery):
    img_path = query.data[8:]
    text = 'Жаль, что вам не понравилось. Мы сделали все возможное! Попробуйте прислать другое фото'
    os.remove(img_path)
    await bot.send_message(query.from_user.id, text)


@dp.callback_query_handler(lambda x: x.data.startswith('no2'), state='*')
async def inline_no2_answer_callback_handler(query: types.CallbackQuery):
    await bot.edit_message_reply_markup(query.message.chat.id, query.message.message_id, None)
    img_name = query.data[4:]
    os.remove(img_name)
    await bot.send_message(query.from_user.id, 'Хорошо. Присылайте фотографии еще!')


@dp.callback_query_handler(lambda x: x.data.startswith('low'), state='*')
async def inline_go_disney_answer_callback_handler(query: types.CallbackQuery):
    img_name = query.data[4:]

    with open(img_name, 'rb') as f:
        await query.message.answer_photo(
            caption='Ваше изображение обрабатывается, это займет примерно {} минут \n'.format(
                recognize_thread.get_waiting_time()
            ),
            photo=f
        )
    await query.message.delete()

    task_queue.put((img_name, query.message.chat.id, low_steps, False))


@dp.callback_query_handler(lambda x: x.data.startswith('high'), state='*')
async def inline_go_disney_answer_callback_handler(query: types.CallbackQuery):
    img_name = query.data[5:]

    with open(img_name, 'rb') as f:
        await query.message.answer_photo(
            caption='Ваше изображение обрабатывается, это займет примерно {} минут \n'.format(
                recognize_thread.get_waiting_time()
            ),
            photo=f
        )
    await query.message.delete()

    task_queue.put((img_name, query.message.chat.id, high_steps, False))


if __name__ == '__main__':
    recognize_thread = RecognizeThread(task_queue)
    recognize_thread.daemon = True
    recognize_thread.start()

    loop = asyncio.get_event_loop()
    loop.create_task(send_ready_images(recognize_thread.ready_queue, bot))

    executor.start_polling(dp, loop=loop)
