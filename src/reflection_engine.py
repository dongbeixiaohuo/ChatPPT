# reflection_engine.py
from typing import List, Dict, Annotated, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from logger import LOG

class ReflectionState(TypedDict):
    """反思状态"""
    messages: List[Dict]        # 对话历史
    current_round: int          # 当前轮数
    optimization_history: Dict   # 优化历史
    should_continue: bool       # 是否继续优化

class ReflectionEngine:
    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.chat_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=4096
        )
        # 初始化工作流图
        self.workflow = self._create_workflow()
    
    def _get_reflection_prompt(self, round_num: int) -> str:
        """获取反思提示"""
        return f"""这是第 {round_num + 1} 轮优化，请基于以下方向优化内容：
        1. 内容深度：补充专业观点、数据支持和研究发现
        2. 案例丰富：增加相关案例、实际应用场景
        3. 结构优化：改进内容组织，使逻辑更清晰
        4. 表达提升：优化语言表达，使内容更生动
        
        请基于上述方向对内容进行优化，保持核心主题不变。"""
    
    def _optimize_content(self, state: ReflectionState) -> ReflectionState:
        """单轮优化节点"""
        try:
            current_round = state["current_round"]
            messages = [
                SystemMessage(content=state["messages"][0]["content"]),  # 系统提示
                HumanMessage(content=state["messages"][-1]["content"]),  # 当前内容
                SystemMessage(content=self._get_reflection_prompt(current_round))
            ]
            
            response = self.chat_model.invoke(messages)
            
            # 更新状态
            state["messages"].append({"role": "assistant", "content": response.content})
            state["optimization_history"][f"round_{current_round}"] = response.content
            state["optimization_history"]["final_content"] = response.content
            state["current_round"] += 1
            
            LOG.info(f"完成第 {current_round + 1} 轮优化")
            return state
            
        except Exception as e:
            LOG.error(f"优化失败: {str(e)}")
            raise

    def _should_continue(self, state: ReflectionState) -> bool:
        """检查是否继续优化"""
        return state["current_round"] < self.max_rounds

    def _create_workflow(self) -> Graph:
        """创建工作流图"""
        # 创建状态图
        workflow = StateGraph(ReflectionState)
        
        # 添加所有节点
        workflow.add_node("optimize", self._optimize_content)
        workflow.add_node("end", lambda x: x)  # 添加终止节点
        
        # 添加条件判断
        workflow.add_conditional_edges(
            "optimize",
            self._should_continue,
            {
                True: "optimize",    # 继续优化
                False: "end"         # 结束优化
            }
        )
        
        # 设置入口点
        workflow.set_entry_point("optimize")
        
        return workflow.compile()

    def optimize_content(self, initial_content: str, system_prompt: str) -> Dict[str, str]:
        """执行多轮优化"""
        try:
            # 初始化状态
            initial_state = ReflectionState(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_content}
                ],
                current_round=0,
                optimization_history={
                    "round_0": initial_content,
                    "final_content": initial_content
                },
                should_continue=True
            )
            
            # 执行工作流
            final_state = self.workflow.invoke(initial_state)
            
            return final_state["optimization_history"]
            
        except Exception as e:
            LOG.error(f"内容优化失败: {str(e)}")
            raise
