import asyncio
from settings import settings
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from langflow import LangFlow

bot = Bot(token=settings.BOT_TOKEN)
dp = Dispatcher()
langflow = LangFlow()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hello!")

@dp.message(F.text)
async def cmd_text(message: types.Message):
    text_message = message.text
    print('CHAT ID: ', message.chat.id)
    print('USER INPUT: ', message.text)

    answer = await langflow.run_flow_chat(message=text_message, session_id=message.chat.id)
    print('MODEL ANSWER: ', answer)

    await message.answer(answer)

@dp.message(F.document)
async def handle_document(message: types.Message):
    doc_file = await bot.get_file(message.document.file_id)
    doc_file_bytes = await bot.download_file(doc_file.file_path)

    content = doc_file_bytes.read().decode('utf-8')

    response = await langflow.load_in_vector_store(message=content, session_id=message.chat.id)
    print('DOWNLOAD FILE RESPONSE: ', response)

    await message.answer(
        f"Документ загружен в векторное хранилище!\n"
    )

async def main():
    await langflow.init_workflow()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
