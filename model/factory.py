# 强制使用国内镜像，解决 huggingface 超时
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from abc import ABC, abstractmethod

from typing import Optional
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from utils.config_handler import rag_conf
from langchain_huggingface import HuggingFaceEmbeddings

class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatDeepSeek(model = rag_conf['chat_model_name'])


class EmbeddingsFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return HuggingFaceEmbeddings(model = rag_conf["embedding_model_name"])

chat_model = ChatModelFactory().generate()
embedding_model = EmbeddingsFactory().generate()