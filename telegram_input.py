import asyncio
import logging
import sys
import os

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from inference import analyze

# настройки
TOKEN = "7894235349:AAHITeKYoFFdC9zKVMz0cns06psVcmdKvAY"
TEMP_FOLDER = "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)


dp = Dispatcher()
router = Router()
dp.include_router(router)

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

#стартовое сообщение
@router.message(CommandStart())
async def start(message: Message):
    await message.answer("Отправьте фото для анализа.")

#обработчик фотографий
@router.message(F.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    photo_path = f"{TEMP_FOLDER}/{photo.file_id}.jpg"
    await bot.download_file(file.file_path, destination=photo_path)

    result = analyze(photo_path)
    await message.answer(f"Результат анализа:\n{result}")


async def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
