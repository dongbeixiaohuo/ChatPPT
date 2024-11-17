# reflection_engine.py
from typing import List, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from logger import LOG

class ReflectionEngine:
    """反思优化引擎，用于多轮优化内容"""
    
    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.chat_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=4096
        )
        
    def _get_reflection_prompt(self, round_num: int) -> str:
        """获取每轮反思的提示"""
        return f"""这是第 {round_num + 1} 轮优化，请基于以下方向优化内容：
        1. 内容深度：补充专业观点、数据支持和研究发现
        2. 案例丰富：增加相关案例、实际应用场景
        3. 结构优化：改进内容组织，使逻辑更清晰
        4. 表达提升：优化语言表达，使内容更生动
        
        请基于上述方向对内容进行优化，保持核心主题不变。"""
    
    def optimize_content(self, initial_content: str, system_prompt: str) -> Dict[str, str]:
        """
        多轮优化内容
        
        Args:
            initial_content: 初始内容
            system_prompt: 系统提示
            
        Returns:
            Dict[str, str]: 包含最终内容和优化过程的字典
        """
        try:
            optimization_history = {
                "round_0": initial_content,
                "final_content": initial_content
            }
            
            current_content = initial_content
            
            for round_num in range(1, self.max_rounds):
                LOG.info(f"开始第 {round_num + 1} 轮优化")
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=current_content),
                    SystemMessage(content=self._get_reflection_prompt(round_num))
                ]
                
                response = self.chat_model.invoke(messages)
                current_content = response.content
                
                optimization_history[f"round_{round_num}"] = current_content
                optimization_history["final_content"] = current_content
                
                LOG.info(f"完成第 {round_num + 1} 轮优化")
            
            return optimization_history
            
        except Exception as e:
            LOG.error(f"内容优化失败: {str(e)}")
            raise
