import gradio as gr
import os
from gradio.data_classes import FileData

from config import Config
from chatbot import ChatBot
from content_formatter import ContentFormatter
from content_assistant import ContentAssistant
from image_advisor import ImageAdvisor
from input_parser import parse_input_text
from ppt_generator import generate_presentation
from template_manager import load_template, get_layout_mapping
from layout_manager import LayoutManager
from logger import LOG
from openai_whisper import asr, transcribe
from docx_parser import generate_markdown_from_docx

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ChatPPT"

# 实例化所有组件
config = Config()
chat_agent = ChatBot(config.chatbot_prompt)
content_formatter = ContentFormatter(config.content_formatter_prompt)
content_assistant = ContentAssistant(config.content_assistant_prompt)
image_advisor = ImageAdvisor(config.image_advisor_prompt)
ppt_template = load_template(config.ppt_template)
layout_manager = LayoutManager(get_layout_mapping(ppt_template))

def generate_contents(message, history):
    """生成幻灯片内容的函数"""
    try:
        # 获取文本输入
        text = message if isinstance(message, str) else message.get("text", "")
        LOG.info(f"[用户输入] {text}")

        if not text:
            raise gr.Error("请输入内容")

        # 获取 chatbot 响应
        messages = chat_agent.chat_with_history(text)
        
        # 确保返回正确的消息格式
        return [
            {"role": "user", "content": text},
            {"role": "assistant", "content": messages[1]["content"]}
        ]

    except gr.Error as e:
        raise e
    except Exception as e:
        LOG.error(f"[内容生成错误]: {e}")
        raise gr.Error(f"网络问题，请重试:)")

def handle_file_upload(file: FileData, history):
    """处理上传的文件"""
    try:
        if not file:
            return history
            
        file_path = file.name
        LOG.debug(f"[上传文件]: {file_path}")
        
        # 获取文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 处理不同类型的文件
        if file_ext in ('.wav', '.flac', '.mp3'):
            # 音频文件转文本
            audio_text = asr(file_path)
            LOG.info(f"[音频转文本]: {audio_text}")
            
            # 获取 chatbot 响应
            messages = chat_agent.chat_with_history(audio_text)
            
            # 返回正确格式的消息
            return [
                {"role": "user", "content": audio_text},
                {"role": "assistant", "content": messages[1]["content"]}
            ]
            
        elif file_ext in ('.docx', '.doc'):
            # Word 文档处理
            raw_content = generate_markdown_from_docx(file_path)
            markdown_content = content_formatter.format(raw_content)
            final_content = content_assistant.adjust_single_picture(markdown_content)
            
            # 返回正确格式的消息
            return [
                {"role": "user", "content": f"处理文件: {os.path.basename(file_path)}"},
                {"role": "assistant", "content": final_content}
            ]
            
        else:
            LOG.debug(f"[格式不支持]: {file_path}")
            raise gr.Error(f"不支持的文件格式: {file_ext}")
            
    except Exception as e:
        LOG.error(f"[文件处理错误]: {e}")
        raise gr.Error(f"文件处理失败，请重试")

def transcribe(audio, history):
    """处理音频转文本"""
    try:
        if not audio:
            return history
            
        # 音频转文本
        text = asr(audio)
        LOG.info(f"[音频转文本]: {text}")
        
        # 获取 chatbot 响应
        messages = chat_agent.chat_with_history(text)
        
        # 返回正确格式的消息
        return [
            {"role": "user", "content": text},
            {"role": "assistant", "content": messages[1]["content"]}
        ]
        
    except Exception as e:
        LOG.error(f"[音频处理错误]: {e}")
        raise gr.Error(f"音频处理失败，请重试")

def handle_image_generate(history):
    """处理图片生成"""
    try:
        # 获取最新的助手回复
        last_message = next((msg for msg in reversed(history) if msg["role"] == "assistant"), None)
        if not last_message:
            raise gr.Error("没有找到可用的内容")
            
        slides_content = last_message["content"]
        content_with_images, image_pair = image_advisor.generate_images(slides_content)
        
        # 返回正确格式的消息
        return [
            {"role": "user", "content": "生成配图"},
            {"role": "assistant", "content": content_with_images}
        ]
    except Exception as e:
        LOG.error(f"[配图生成错误]: {e}")
        raise gr.Error(f"【提示】未找到合适配图，请重试！")

def handle_generate(history):
    """处理 PPT 生成"""
    try:
        slides_content = history[-1][1]  # 获取最新的助手回复
        powerpoint_data, presentation_title = parse_input_text(slides_content, layout_manager)
        output_pptx = f"outputs/{presentation_title}.pptx"
        
        generate_presentation(powerpoint_data, config.ppt_template, output_pptx)
        return output_pptx
    except Exception as e:
        LOG.error(f"[PPT 生成错误]: {e}")
        raise gr.Error(f"【提示】请先输入你的主题内容或上传文件")

# 创建 Gradio 界面
with gr.Blocks(title="ChatPPT") as demo:
    gr.Markdown("## ChatPPT")

    # 创建聊天机器人界面
    chat_interface = gr.Chatbot(
        value=[],
        height=800,
        type="messages",
        avatar_images=["user.png", "assistant.png"]
    )
    
    # 输入区域
    with gr.Row():
        # 文本输入
        with gr.Column(scale=3):
            msg = gr.Textbox(
                show_label=False,
                placeholder="输入你的主题内容...",
                container=False
            )
        
        # 语音输入
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources="microphone",
                type="filepath",
                label="语音输入"
            )
            
        # 文件上传
        with gr.Column(scale=1):
            file_input = gr.File(
                label="上传文件",
                file_types=[".wav", ".mp3", ".flac", ".docx", ".doc"]
            )
            
        # 清除按钮
        with gr.Column(scale=1, min_width=50):
            clear = gr.Button("清除")
    
    # 功能按钮
    with gr.Row():
        image_generate_btn = gr.Button("一键为 PowerPoint 配图")
        generate_btn = gr.Button("一键生成 PowerPoint")
    
    # 绑定事件
    msg.submit(generate_contents, [msg, chat_interface], [chat_interface])
    audio_input.change(transcribe, [audio_input, chat_interface], [chat_interface])
    file_input.upload(handle_file_upload, [file_input, chat_interface], [chat_interface])
    clear.click(lambda: None, None, chat_interface, queue=False)
    
    image_generate_btn.click(
        fn=handle_image_generate,
        inputs=chat_interface,
        outputs=chat_interface,
    )

    generate_btn.click(
        fn=handle_generate,
        inputs=chat_interface,
        outputs=gr.File()
    )

# 启动应用
if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        server_port=17860,
        server_name="0.0.0.0",
        auth=("user", "Linemore"),
        debug=True
    )
