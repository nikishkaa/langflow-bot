import asyncio
import json
from traceback import print_tb

from aiohttp import ClientSession, ClientError

from settings import settings

class LangFlow:
    def __init__(self):
        self.url = settings.lagflow_host
        self.chat_bot_config_path = 'fixtures/Chat-bot.json'
        self.store_config_path = 'fixtures/Store.json'
        self.run_flow_url = f"{self.url}/api/v1/run/"
        self.create_flow_url = f"{self.url}/api/v1/flows/"
        self.chat_chain_id = None
        self.store_chain_id = None

    async def init_workflow(self):
        await self._waiter()
        self.chat_chain_id = await self._load_workflow(self.chat_bot_config_path)
        self.store_chain_id = await self._load_workflow(self.store_config_path)
        await self.init_context()

    async def init_context(self):
        print('Началась загрузка контекст файла...')
        with open(settings.CONTEXT_FILE_PATH) as context_file:
            context = context_file.read()
        response = await self.load_in_vector_store(context, session_id=0)
        print('Контекст файл загружен!')

    async def run_flow_chat(self, message: str, session_id: int) -> str:
        url = self.run_flow_url + self.chat_chain_id
        payload = {
            "input_value": message,
            "output_type": "text",
            "input_type": "chat",
            "session_id": str(session_id),
            "tweaks": {
                "OllamaEmbeddings-kjmsv": {
                    "base_url": settings.ollama_host,
                    "model_name": settings.OLLAMA_EMBEDDING_MODEL
                },
                "OllamaModel-EKIv8": {
                    "base_url": settings.ollama_host,
                    "model_name": settings.OLLAMA_LLM
                },
                "Chroma-imobu": {
                    "persist_directory": settings.CHROMA_DIR
                }
            }
        }

        response = await self._post_query_langflow(url, payload)

        print('LANGFLOW RESPONSE: ', response)

        try:
            outputs = response.get('outputs')
            outputs_2 = outputs[0].get('outputs')
            results = outputs_2[0].get('results')
            message_tg = results.get('message')
            text = message_tg.get('text')
        except Exception:
            raise Exception('Невалидный ответ из langflow...')

        return text

    async def load_in_vector_store(self, message: str, session_id: int) -> dict:
        url = self.run_flow_url + self.store_chain_id

        payload = {
            "input_value": message,
            "output_type": "text",
            "input_type": "chat",
            "tweaks": {
                "OllamaEmbeddings-pM407": {
                    "base_url": settings.ollama_host,
                    "model_name": settings.OLLAMA_EMBEDDING_MODEL
                },
                "Chroma-0TsX2": {
                    "persist_directory": settings.CHROMA_DIR
                },
                "RecursiveCharacterTextSplitter-mdQcv":{
                    "separators": settings.SEPARATORS,
                    "chunk_size": settings.CHUNK_SIZE,
                }
            }
        }

        if session_id != 0:
            payload['session_id'] = str(session_id)

        response = await self._post_query_langflow(url, payload)
        return response

    async def _load_workflow(self, config_path: str) -> str:
        with open(config_path, 'r') as chat_bot_config_file:
            chat_bot_config = json.load(chat_bot_config_file)

        response = await self._post_query_langflow(self.create_flow_url, chat_bot_config)

        print(f'Create flow with id: {response.get('id')}')
        return response.get('id')

    @staticmethod
    async def _post_query_langflow(url, payload) -> dict | None:
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        async with ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                response = await resp.json()
        return response

    async def _waiter(self):
        while True:
            try:
                async with ClientSession() as session:
                    async with session.get(self.url) as response:
                        if response.status == 200:
                            print("Connection to Langflow successful!")
                            break
            except ClientError as e:
                print(f"Connection to Langflow failed: {e}. Next try...")
            await asyncio.sleep(5)
