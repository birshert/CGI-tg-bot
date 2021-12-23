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

start_steps = 300
step_of_steps = 200
end_steps = 1500
iter_crop = 70


@dp.message_handler(commands=['start', 'help'])
async def some_handler(message: types.Message):
    start_message = 'Привет! Это бот, который умеет преобразовывать фотографии лиц в мультяшные. \n' \
                    'Просто пришли нам свою фотку, и мы сделаем из тебя диснеевскую принцессу)'
    print(message.chat.id)
    await message.answer(start_message)


@dp.message_handler(content_types=['photo'], state='*')
async def handle_docs_photo(message):
    document_id = message['photo'][-1].file_id
    file_id = message['photo'][-1].file_unique_id

    file_info = await bot.get_file(document_id)
    filename, file_extension = os.path.splitext(file_info.file_path)
    image_name = os.path.join('photos', file_id + file_extension)
    urllib.request.urlretrieve(f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}', image_name)
    recognize_thread.iter += iter_crop
    task_queue.put((image_name, message.chat.id, start_steps, True))
    await message.answer('Ищем лица на вашем фото. Ожидание составит примерно ' +
                         recognize_thread.get_waiting_time() + '\n')


@dp.callback_query_handler(lambda x: x.data.startswith('yes'), state='*')
async def inline_yes1_answer_callback_handler(query: types.CallbackQuery):
    await bot.edit_message_reply_markup(query.message.chat.id, query.message.message_id, None)
    text = 'Отлично! Мы рады, что вам понравилось! Присылайте фотографии еще'
    img_path = query.data.split('|')[1]
    os.remove(img_path)
    os.remove(img_path.split('.')[0] + '.npz')
    await bot.send_message(query.from_user.id, text)


@dp.callback_query_handler(lambda x: x.data.startswith('do'), state='*')
async def inline_go_disney_answer_callback_handler1(query: types.CallbackQuery):
    img_name = query.data.split('|')[2]
    if int(query.data.split('|')[1]) == start_steps:
        recognize_thread.iter += start_steps
    else:
        recognize_thread.iter += step_of_steps
    nm = int(query.data.split('|')[1])
    await bot.edit_message_reply_markup(query.message.chat.id, query.message.message_id, None)
    if nm <= end_steps:
        await query.message.edit_caption('Сочи. Сириус. Работаем. Ожидание составит примерно ' +
                                         recognize_thread.get_waiting_time() + '\n')
        task_queue.put((img_name, query.message.chat.id, int(query.data.split('|')[1]), False))
    else:
        os.remove(img_name)
        os.remove(img_name.split('.')[0] + '.npz')
        await bot.send_message(chat_id=query.message.chat.id,
                               text="Мы сделали все возможное. Попробуйте прислать другое фото.")


if __name__ == '__main__':
    recognize_thread = RecognizeThread(task_queue)
    recognize_thread.daemon = True
    recognize_thread.start()

    loop = asyncio.get_event_loop()
    loop.create_task(send_ready_images(recognize_thread.ready_queue, bot))

    executor.start_polling(dp, loop=loop)
