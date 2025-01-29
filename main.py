import streamlit as st
from openai import OpenAI
import PyPDF2
from io import BytesIO
import fitz  # PyMuPDF
from fuzzywuzzy import fuzz  # 用于模糊匹配
import json
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import rapidfuzz
from rapidfuzz import fuzz
from concurrent.futures import ThreadPoolExecutor

# Streamlit应用
st.title("🎨 HighlightScholar")

# 用户输入API Key和Base URL
api_key = st.secrets["api_key"] #st.text_input("请输入您的OpenAI API Key", type="password")
base_url = st.secrets["base_url"] #st.text_input("请输入您的OpenAI Base URL", )


# 初始化OpenAI客户端
client = OpenAI(api_key=api_key, base_url=base_url)
#if api_key and base_url:
#    client = OpenAI(api_key=api_key, base_url=base_url)
#else:
#    st.warning("请输入API Key和Base URL以继续。")
#    st.stop()

# 从PDF中提取文本
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text, reader

# 压缩文本
def compress_text(text, target_sentences=60):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=target_sentences)
    compressed_text = ''
    for s in summary:
        compressed_text += ' '.join(s.words) + '. '
    return compressed_text

# 使用OpenAI GPT模型提取关键内容
def extract_key_content(text):
    prompt = f"""
    请从以下学术文章中提取以下关键内容，并以严格的JSON格式返回：
    {{
        "Background": ["直接引用原文中描述研究背景的句子1。", "直接引用原文中描述研究背景的句子2。" ...],
        "Research Gap": ["直接引用原文中描述已有研究局限的句子1。", "直接引用原文中描述已有研究局限的句子2。" ...],
        "Research Questions/Hypotheses": ["直接引用原文中描述研究问题或假设的句子1。", "直接引用原文中描述研究问题或假设的句子2。","直接引用原文中描述研究问题或假设的句子3。", "直接引用原文中描述研究问题或假设的句子4。", "直接引用原文中描述研究问题或假设的句子5。"...],
        "Research Method": ["直接引用原文中描述研究方法的句子1。", "直接引用原文中描述研究方法的句子2。", "直接引用原文中描述研究方法的句子3。"...],
        "Research Data": ["直接引用原文中描述数据收集的句子1。", "直接引用原文中描述数据收集的句子2。" ...],
        "Main Findings": ["直接引用原文中描述研究发现的句子1。", "直接引用原文中描述研究发现的句子2。", "直接引用原文中描述研究发现的句子3。",...],
        "Originality/Innovation": ["直接引用原文中描述研究创新点的句子1。", "直接引用原文中描述研究创新点的句子2。" ...]
    }}

    注意：
    1. 请确保返回的内容是原文中的直接引用，不要改变原文的语言和内容形式。
    2. 确保返回的内容是严格的JSON格式，键和值都用双引号括起来。
    3. 每个关键内容的值是一个包含多个句子的列表。
    4. 请保证关键内容被全面提取，每个关键内容的列表中值的数量为1到10个。
    5. 避免不同关键内容中出现重复的句子。

    文章内容：
    {text}
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个学术助手，能够从文章中提取关键内容。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2
    )
    
    # 提取返回的内容
    result = response.choices[0].message.content.strip()
    return result

def fuzzy_match_sentence(sentences, keyword_list, threshold=55):
    # 预处理句子和关键词
    sentences_clean = [re.sub(r'[^\w\s]', '', s.lower()) for s in sentences]
    keyword_list_clean = [re.sub(r'[^\w\s]', '', k.lower()) for k in keyword_list]
    
    # 用于存储匹配的句子
    matched_sentences = set()  # 使用集合避免重复
    
    # 定义一个函数，用于匹配单个关键词
    def match_keyword(keyword_clean):
        matched = []
        for i, sentence_clean in enumerate(sentences_clean):
            # 提前过滤：如果句子长度与关键词长度差异过大，跳过
            if abs(len(sentence_clean) - len(keyword_clean)) > 20:
                continue
            # 计算相似度
            similarity = fuzz.token_sort_ratio(sentence_clean, keyword_clean)
            if similarity >= threshold:
                matched.append(sentences[i])  # 使用原始句子
        return matched
    
    # 使用多线程并行处理关键词
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(match_keyword, keyword_clean) for keyword_clean in keyword_list_clean]
        for future in futures:
            matched_sentences.update(future.result())
    
    return list(matched_sentences)

# 在PDF中高亮关键内容
def highlight_pdf(pdf_bytes, key_content, text):
    # 打开PDF文件（直接从内存中读取）
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # 解析提取的关键内容
    try:
        key_content = key_content.strip('```json').strip()
        key_content_dict = json.loads(key_content)
    except json.JSONDecodeError as e:
        st.error(f"提取的关键内容格式不正确，无法高亮。错误信息：{e}")
        return None
    
    
    # 定义不同关键内容的高亮颜色
    highlight_colors = {
        "Background": (0.95, 0.6, 0.6),  # 柔和的红色
        "Research Gap": (1.0, 0.7, 0.4),  # 柔和的橙色
        "Research Questions/Hypotheses": (0.6, 0.95, 0.6),  # 柔和的绿色
        "Research Method": (0.95, 0.95, 0.6),  # 柔和的黄色
        "Research Data": (0.6, 0.6, 0.95),  # 柔和的蓝色
        "Main Findings": (0.6, 0.95, 0.95),  # 柔和的青色
        "Originality/Innovation": (0.95, 0.6, 0.95),  # 柔和的紫色
    }
    
    # 在第一页添加色块和注释
    first_page = doc.load_page(0)
    page_width = first_page.rect.width
    page_height = first_page.rect.height
    
    # 色块和注释的起始位置
    x0 = page_width - 150  # 色块起始X坐标
    y0 = 10  # 色块起始Y坐标
    block_height = 5  # 每个色块的高度
    spacing = 5  # 色块之间的间距
    
    for key, color in highlight_colors.items():
        # 绘制色块
        rect = fitz.Rect(x0, y0, x0 + 30, y0 + block_height)
        first_page.draw_rect(rect, color=color, fill=color)

        # 添加文本
        text_point = (x0 + 40, y0 + block_height / 2 + 5)  # 文本位置（色块右侧居中）
        first_page.insert_text(
            text_point,  # 文本插入位置
            key,  # 文本内容
            fontsize=6,  # 字体大小
            fontname="helv",  # 字体名称
            color=(0, 0, 0)  # 文本颜色（黑色）
        )

        # 更新下一个色块的Y坐标
        y0 += block_height + spacing
    
    # 遍历每一页
    for page_num in range(len(doc)):
        
        # 初始化一个list，用于记录已经高亮的句子
        highlighted_sentences = []
    
        page = doc.load_page(page_num)
        page_text = page.get_text().replace('\n', ' ').replace('  ', ' ')  # 获取当前页的文本
        
        # 将文本按句子分割
        sentences = re.split(r'[.!?]', page_text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
        if not page_text.strip():  # 检查页面文本是否为空
            st.warning(f"第 {page_num + 1} 页的文本不可搜索，可能是图像或扫描件。")
            continue
        
        # 遍历关键内容并高亮
        for key, value in key_content_dict.items():
            if isinstance(value, list):  # 确保值是列表
                # 使用句子级别的模糊匹配查找相似句子
                matched_sentences = list(set(fuzzy_match_sentence(sentences, value)))
                
                for sentence in matched_sentences:
                    if len(sentence.split(' ')) < 5:
                        continue 
                    
                    if sentence not in highlighted_sentences:
                        highlighted_sentences.append(sentence)
                    
                        # 查找匹配句子在页面中的位置
                        text_instances = page.search_for(sentence)

                        for inst in text_instances:
                            # 添加高亮注释，并使用不同的颜色
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=highlight_colors.get(key, (1, 1, 0)))  # 默认黄色
                            highlight.update()
    
    # 将高亮后的PDF保存到内存中
    output_bytes = BytesIO()
    doc.save(output_bytes)
    doc.close()
    output_bytes.seek(0)
    
    return output_bytes

# 上传PDF文件
uploaded_file = st.file_uploader("上传学术文章 (PDF格式)", type="pdf")
if uploaded_file is not None:
    # 将上传的文件读取到内存中
    pdf_bytes = uploaded_file.read()
    
    # 提取文本
    text, pdf_reader = extract_text_from_pdf(BytesIO(pdf_bytes))
    
    # 显示提取的文本
    st.write("### 提取的文本")
    st.write(text[:5000] + "...")  # 显示前5000字符（避免页面过长）

    # 添加 "Highlight" 按钮
    if st.button("Highlight"):
        with st.spinner("正在提取关键内容并高亮..."):
            # 提取关键内容
            key_content = extract_key_content(text)
            st.write("### 提取的关键内容")
            st.write(key_content)

            # 高亮PDF中的关键内容
            highlighted_pdf = highlight_pdf(pdf_bytes, key_content, text)
            
            if highlighted_pdf:
                # 提供下载链接
                st.download_button(
                    label="下载高亮PDF",
                    data=highlighted_pdf.getvalue(),  # 获取字节流的值
                    file_name="highlighted_article.pdf",
                    mime="application/pdf"
                )
                
                # 预览高亮PDF
                st.write("### 高亮PDF预览")
                st.markdown("由于Streamlit不支持直接预览PDF文件，请下载后查看高亮效果。")
