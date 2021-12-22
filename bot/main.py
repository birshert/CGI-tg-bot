import asyncio
import os
import queue
import urllib

from aiogram import Bot
from aiogram import types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import Dispatcher
from aiogram.types import InlineKeyboardButton
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils import executor
from aiogram.utils.helper import Helper
from aiogram.utils.helper import HelperMode
from aiogram.utils.helper import ListItem

from config import BOT_TOKEN
from wokrer import RecognizeThread
from wokrer import send_ready_images


bot = Bot(token=BOT_TOKEN)

dp = Dispatcher(bot, storage=MemoryStorage())
task_queue = queue.Queue()


class States(Helper):
    mode = HelperMode.snake_case

    STATE_0 = ListItem()
    STATE_1 = ListItem()
    STATE_2 = ListItem()


@dp.message_handler(commands=['start', 'help'])
async def some_handler(message: types.Message):
    start_message = "Привет! Это бот, который умеет преобразовывать фотографии лиц в мультяшные. \n" \
                    "Просто пришли нам свою фотку, и мы сделаем из тебя диснеевскую принцессу)"
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(States.all()[0])
    await message.answer(start_message)


@dp.message_handler(content_types=['photo'], state='*')
async def handle_docs_photo(message):
    document_id = message['photo'][-1].file_id
    state = dp.current_state(user=message.from_user.id)
    file_id = message['photo'][-1].file_unique_id

    file_info = await bot.get_file(document_id)
    filename, file_extension = os.path.splitext(file_info.file_path)
    image_name = os.path.join('photos', file_id + file_extension)
    urllib.request.urlretrieve(f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}', image_name)
    if await state.get_state() == States.STATE_0[0]:
        num_steps = 300
    elif await state.get_state() == States.STATE_1[0]:
        num_steps = 500
    elif await state.get_state() == States.STATE_2[0]:
        num_steps = 1000
    else:
        await state.set_state(States.all()[0])
        num_steps = 300
    task_queue.put((image_name, message.chat.id, num_steps, True))
    await message.answer(
        "Ищем лица на вашем фото")


@dp.callback_query_handler(lambda x: x.data.startswith('yes1'), state='*')
async def inline_yes1_answer_callback_handler(query: types.CallbackQuery):
    await bot.edit_message_reply_markup(query.message.chat.id, query.message.message_id, None)
    text = "Отлично! Мы рады, что вам понравилось! Присылайте фотографии еще"
    state = dp.current_state(user=query.from_user.id)
    img_path = query.data[5:]
    os.remove(img_path)
    await state.set_state(States.all()[0])
    await bot.send_message(query.from_user.id, text)


@dp.callback_query_handler(lambda x: x.data.startswith('no1'), state='*')
async def inline_no1_answer_callback_handler(query: types.CallbackQuery):
    await bot.edit_message_reply_markup(query.message.chat.id, query.message.message_id, None)
    state = dp.current_state(user=query.from_user.id)
    img_path = query.data[4:]
    if await state.get_state() == States.STATE_2[0]:
        text = "Жаль, что вам не понравилось. Мы сделали все возможное! Попробуйте прислать другое фото"
        os.remove(img_path)
        await state.set_state(States.all()[0])
        await bot.send_message(query.from_user.id, text)
    else:
        text = "Хотите ли вы, чтобы мы обработали последнюю фотографию чуть дольше, тем самым улучшив результат?"
        markup = InlineKeyboardMarkup()
        markup.row_width = 2
        markup.add(InlineKeyboardButton("Да", callback_data="yes2_" + img_path),
                   InlineKeyboardButton("Нет", callback_data="no2_" + img_path))

        await query.message.answer(text, reply_markup=markup)


@dp.callback_query_handler(lambda x: x.data.startswith('yes2'), state='*')
async def inline_yes2_answer_callback_handler(query: types.CallbackQuery):
    await bot.edit_message_reply_markup(query.message.chat.id, query.message.message_id, None)
    state = dp.current_state(user=query.from_user.id)
    if await state.get_state() == States.STATE_0[0]:
        await state.set_state(States.all()[1])
    elif await state.get_state() == States.STATE_1[0]:
        await state.set_state(States.all()[2])
    img_name = query.data[5:]
    num_steps = 300
    if await state.get_state() == States.STATE_0[0]:
        num_steps = 300
    if await state.get_state() == States.STATE_1[0]:
        num_steps = 500
    if await state.get_state() == States.STATE_2[0]:
        num_steps = 1000
    task_queue.put((img_name, query.message.chat.id, num_steps, False))
    await query.message.answer("Ваше изображение обрабатывается, это займет примерно {} минут \n".format(
        recognize_thread.get_waiting_time()))


@dp.callback_query_handler(lambda x: x.data.startswith('no2'), state='*')
async def inline_yes2_answer_callback_handler(query: types.CallbackQuery):
    await bot.edit_message_reply_markup(query.message.chat.id, query.message.message_id, None)
    state = dp.current_state(user=query.from_user.id)
    img_name = query.data[4:]
    os.remove(img_name)
    await state.set_state(States.all()[0])
    await bot.send_message(query.from_user.id, "Хорошо. Присылайте фотографии еще!")


@dp.callback_query_handler(lambda x: x.data.startswith('yes0'), state='*')
async def inline_go_disney_answer_callback_handler(query: types.CallbackQuery):
    await bot.edit_message_reply_markup(query.message.chat.id, query.message.message_id, None)
    img_name = query.data[5:]
    task_queue.put((img_name, query.message.chat.id, 100, False))
    await query.message.answer("Ваше изображение обрабатывается, это займет примерно {} минут \n".format(
        recognize_thread.get_waiting_time()))


if __name__ == '__main__':
    recognize_thread = RecognizeThread(task_queue)
    recognize_thread.daemon = True
    recognize_thread.start()

    loop = asyncio.get_event_loop()
    loop.create_task(send_ready_images(recognize_thread.ready_queue, bot))

    executor.start_polling(dp, loop=loop)
