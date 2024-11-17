# chatbot.py

from abc import ABC, abstractmethod
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入提示模板相关类
from langchain_core.messages import HumanMessage  # 导入消息类
from langchain_core.runnables.history import RunnableWithMessageHistory  # 导入带有消息历史的可运行类
from openai import OpenAI  # 确保导入 OpenAI

from logger import LOG  # 导入日志工具
from chat_history import get_session_history
from reflection_engine import ReflectionEngine  # 导入反思引擎


class ChatBot(ABC):
    """
    聊天机器人基类，提供聊天功能。
    """
    def __init__(self, prompt_file="./prompts/chatbot.txt", session_id=None):
        self.prompt_file = prompt_file
        self.session_id = session_id if session_id else "default_session_id"
         # 设置 OpenAI API 配置
        os.environ["OPENAI_API_BASE"] = "https://api.javis3000.com/v1"
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        self.prompt = self.load_prompt()
        # LOG.debug(f"[ChatBot Prompt]{self.prompt}")
        self.create_chatbot()
        # 初始化反思引擎
        self.reflection_engine = ReflectionEngine()

    def load_prompt(self):
        """
        从文件加载系统提示语。
        """
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到提示文件 {self.prompt_file}!")


    def create_chatbot(self):
        """
        初始化聊天机器人，包括系统提示和消息历史记录。
        """
        # 创建聊天提示模板，包括系统提示和消息占位符
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),  # 系统提示部分
            MessagesPlaceholder(variable_name="messages"),  # 消息占位符
        ])

        # 初始化 ChatOllama 模型，配置参数
        self.chatbot = system_prompt | ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=4096
        )

        # 将聊天机器人与消息历史记录关联
        self.chatbot_with_history = RunnableWithMessageHistory(self.chatbot, get_session_history)


    def chat_with_history(self, text: str) -> str:
        """
        与 AI 对话并返回回复内容
        适配 ChatInterface 的格式要求
        """
        try:
            # 使用反思引擎生成优化后的内容
            optimization_result = self.reflection_engine.optimize_content(
                initial_content=text,
                system_prompt=self.prompt
            )
            
            # 获取最终优化后的内容
            final_content = optimization_result["final_content"]
            LOG.debug(f"[AI Response] {final_content}")
            
            # 返回文本内容
            return final_content

        except Exception as e:
            LOG.error(f"[Chat Error] {e}")
            raise e

