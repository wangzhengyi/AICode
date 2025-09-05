#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统 - 基于DeepSeek模型的检索增强生成系统

本文件实现了一个完整的RAG（Retrieval-Augmented Generation）系统，包括：
1. 文档分块处理
2. 文本向量化
3. ChromaDB向量数据库存储
4. 语义检索
5. 结果重排序
6. 基于DeepSeek模型的答案生成

作者: AI Assistant
日期: 2024
"""

import os
import logging
import time
from typing import List, Tuple, Optional
from pathlib import Path

# 第三方库导入
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ 缺少必要的依赖包: {e}")
    print("请运行: uv add sentence-transformers chromadb openai python-dotenv")
    exit(1)


class RAGLogger:
    """RAG系统日志管理器
    
    提供统一的日志记录功能，支持不同级别的日志输出
    包括控制台输出和文件记录
    """
    
    def __init__(self, log_file: str = "rag_system.log", log_level: int = logging.INFO):
        """
        初始化日志系统
        
        Args:
            log_file: 日志文件路径
            log_level: 日志级别
        """
        self.logger = logging.getLogger("RAGSystem")
        self.logger.setLevel(log_level)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # 文件处理器
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, emoji: str = "ℹ️"):
        """记录信息级别日志"""
        self.logger.info(f"{emoji} {message}")
    
    def success(self, message: str):
        """记录成功信息"""
        self.logger.info(f"✅ {message}")
    
    def warning(self, message: str):
        """记录警告信息"""
        self.logger.warning(f"⚠️ {message}")
    
    def error(self, message: str):
        """记录错误信息"""
        self.logger.error(f"❌ {message}")
    
    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(f"🔍 {message}")


class DocumentProcessor:
    """文档处理器
    
    负责将长文档分割成适合向量化的文本块
    支持多种分割策略和重叠处理
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, logger: Optional[RAGLogger] = None):
        """
        初始化文档处理器
        
        Args:
            chunk_size: 每个文本块的最大字符数
            overlap: 相邻文本块之间的重叠字符数
            logger: 日志记录器
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logger or RAGLogger()
        
        self.logger.info(f"文档处理器初始化完成 - 块大小: {chunk_size}, 重叠: {overlap}")
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成多个块
        
        Args:
            text: 待分割的文本
            
        Returns:
            分割后的文本块列表
        """
        if not text.strip():
            self.logger.warning("输入文本为空")
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # 计算当前块的结束位置
            end = min(start + self.chunk_size, text_length)
            
            # 提取当前块
            chunk = text[start:end].strip()
            
            if chunk:  # 只添加非空块
                chunks.append(chunk)
            
            # 计算下一个块的开始位置（考虑重叠）
            next_start = end - self.overlap
            
            # 避免无限循环：确保下一个开始位置向前推进
            if next_start <= start:
                next_start = start + max(1, self.chunk_size - self.overlap)
            
            start = next_start
            
            # 如果剩余文本太短，直接退出
            if start >= text_length:
                break
        
        self.logger.success(f"文本分割完成 - 原文长度: {text_length}, 生成块数: {len(chunks)}")
        return chunks
    
    def load_document(self, file_path: str) -> str:
        """
        从文件加载文档内容
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            文档内容字符串
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.logger.success(f"文档加载成功 - 文件: {file_path}, 长度: {len(content)}")
            return content
            
        except Exception as e:
            self.logger.error(f"文档加载失败: {e}")
            raise


class VectorEmbedder:
    """向量嵌入器
    
    使用SentenceTransformer模型将文本转换为向量表示
    支持中文文本的语义向量化
    """
    
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese", logger: Optional[RAGLogger] = None):
        """
        初始化向量嵌入器
        
        Args:
            model_name: 预训练模型名称
            logger: 日志记录器
        """
        self.model_name = model_name
        self.logger = logger or RAGLogger()
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """
        加载预训练的向量化模型
        """
        try:
            self.logger.info(f"正在加载向量化模型: {self.model_name}")
            start_time = time.time()
            
            # 加载模型
            self.model = SentenceTransformer(self.model_name)
            
            load_time = time.time() - start_time
            self.logger.success(f"模型加载完成 - 耗时: {load_time:.2f}秒")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        将单个文本转换为向量
        
        Args:
            text: 待向量化的文本
            
        Returns:
            文本的向量表示
        """
        if not self.model:
            raise RuntimeError("模型未加载")
        
        try:
            # 生成向量并归一化
            embedding = self.model.encode(text, normalize_embeddings=True)
            
            self.logger.debug(f"文本向量化完成 - 文本长度: {len(text)}, 向量维度: {len(embedding)}")
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"文本向量化失败: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量向量化文本列表
        
        Args:
            texts: 待向量化的文本列表
            
        Returns:
            文本向量列表
        """
        if not self.model:
            raise RuntimeError("模型未加载")
        
        try:
            self.logger.info(f"开始批量向量化 - 文本数量: {len(texts)}")
            start_time = time.time()
            
            # 批量生成向量并归一化
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            
            process_time = time.time() - start_time
            self.logger.success(f"批量向量化完成 - 耗时: {process_time:.2f}秒")
            
            return embeddings.tolist()
            
        except Exception as e:
            self.logger.error(f"批量向量化失败: {e}")
            raise


class ChromaDBManager:
    """ChromaDB向量数据库管理器
    
    负责向量的存储、检索和管理
    提供高效的语义搜索功能
    """
    
    def __init__(self, collection_name: str = "rag_documents", logger: Optional[RAGLogger] = None):
        """
        初始化ChromaDB管理器
        
        Args:
            collection_name: 集合名称
            logger: 日志记录器
        """
        self.collection_name = collection_name
        self.logger = logger or RAGLogger()
        self.client = None
        self.collection = None
        
        self._initialize_db()
    
    def _initialize_db(self):
        """
        初始化ChromaDB客户端和集合
        """
        try:
            self.logger.info("正在初始化ChromaDB")
            
            # 创建内存客户端（用于演示，生产环境建议使用持久化客户端）
            self.client = chromadb.EphemeralClient()
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG系统文档向量存储"}
            )
            
            self.logger.success(f"ChromaDB初始化完成 - 集合: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"ChromaDB初始化失败: {e}")
            raise
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], ids: Optional[List[str]] = None):
        """
        向数据库添加文档和对应的向量
        
        Args:
            texts: 文档文本列表
            embeddings: 对应的向量列表
            ids: 文档ID列表（可选）
        """
        if not self.collection:
            raise RuntimeError("ChromaDB集合未初始化")
        
        if len(texts) != len(embeddings):
            raise ValueError("文本数量与向量数量不匹配")
        
        try:
            # 生成ID（如果未提供）
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(texts))]
            
            self.logger.info(f"正在添加文档到数据库 - 数量: {len(texts)}")
            
            # 添加到集合
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids
            )
            
            self.logger.success(f"文档添加完成 - 总数: {self.collection.count()}")
            
        except Exception as e:
            self.logger.error(f"文档添加失败: {e}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> Tuple[List[str], List[float]]:
        """
        基于向量进行语义搜索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回的最相关文档数量
            
        Returns:
            (相关文档列表, 相似度分数列表)
        """
        if not self.collection:
            raise RuntimeError("ChromaDB集合未初始化")
        
        try:
            self.logger.info(f"正在执行语义搜索 - top_k: {top_k}")
            
            # 执行查询
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            self.logger.success(f"搜索完成 - 找到 {len(documents)} 个相关文档")
            
            return documents, distances
            
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            raise
    
    def get_collection_info(self) -> dict:
        """
        获取集合信息
        
        Returns:
            集合统计信息
        """
        if not self.collection:
            return {"error": "集合未初始化"}
        
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "status": "active"
            }
        except Exception as e:
            self.logger.error(f"获取集合信息失败: {e}")
            return {"error": str(e)}


class ResultReranker:
    """结果重排序器
    
    使用CrossEncoder模型对检索结果进行重新排序
    提高检索结果的相关性和准确性
    """
    
    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", logger: Optional[RAGLogger] = None):
        """
        初始化重排序器
        
        Args:
            model_name: CrossEncoder模型名称
            logger: 日志记录器
        """
        self.model_name = model_name
        self.logger = logger or RAGLogger()
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """
        加载CrossEncoder模型
        """
        try:
            self.logger.info(f"正在加载重排序模型: {self.model_name}")
            start_time = time.time()
            
            self.model = CrossEncoder(self.model_name)
            
            load_time = time.time() - start_time
            self.logger.success(f"重排序模型加载完成 - 耗时: {load_time:.2f}秒")
            
        except Exception as e:
            self.logger.error(f"重排序模型加载失败: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[str]:
        """
        对检索结果进行重新排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的文档数量（可选）
            
        Returns:
            重新排序后的文档列表
        """
        if not self.model:
            raise RuntimeError("重排序模型未加载")
        
        if not documents:
            self.logger.warning("没有文档需要重排序")
            return []
        
        try:
            self.logger.info(f"正在重排序 - 文档数量: {len(documents)}")
            
            # 创建查询-文档对
            pairs = [(query, doc) for doc in documents]
            
            # 计算相关性分数
            scores = self.model.predict(pairs)
            
            # 按分数排序
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # 提取排序后的文档
            reranked_docs = [doc for doc, _ in scored_docs]
            
            # 限制返回数量
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]
            
            self.logger.success(f"重排序完成 - 返回 {len(reranked_docs)} 个文档")
            
            return reranked_docs
            
        except Exception as e:
            self.logger.error(f"重排序失败: {e}")
            raise


class DeepSeekGenerator:
    """DeepSeek生成器
    
    使用DeepSeek API进行文本生成
    基于检索到的相关文档生成准确的答案
    """
    
    def __init__(self, logger: Optional[RAGLogger] = None):
        """
        初始化DeepSeek生成器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or RAGLogger()
        self.client = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """
        初始化DeepSeek API客户端
        """
        try:
            # 加载环境变量
            load_dotenv()
            
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            
            if not api_key:
                raise ValueError("未找到DEEPSEEK_API_KEY环境变量")
            
            self.logger.info("正在初始化DeepSeek客户端")
            
            # 创建OpenAI客户端（兼容DeepSeek API）
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            self.logger.success("DeepSeek客户端初始化完成")
            
        except Exception as e:
            self.logger.error(f"DeepSeek客户端初始化失败: {e}")
            raise
    
    def generate_answer(self, query: str, context_documents: List[str], model: str = "deepseek-chat") -> str:
        """
        基于查询和上下文文档生成答案
        
        Args:
            query: 用户查询
            context_documents: 相关上下文文档
            model: 使用的模型名称
            
        Returns:
            生成的答案
        """
        if not self.client:
            raise RuntimeError("DeepSeek客户端未初始化")
        
        try:
            self.logger.info(f"正在生成答案 - 查询: {query[:50]}...")
            
            # 构建提示词
            context = "\n\n".join(context_documents)
            prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:
{context}

请基于上述内容作答，不要编造信息。"""
            
            # 调用DeepSeek API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # 提取生成的内容
            answer = response.choices[0].message.content
            
            self.logger.success(f"答案生成完成 - 长度: {len(answer)}")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"答案生成失败: {e}")
            raise


class RAGSystem:
    """RAG系统主类
    
    整合所有组件，提供完整的RAG功能
    包括文档处理、向量化、存储、检索和生成
    """
    
    def __init__(self, 
                 chunk_size: int = 500,
                 overlap: int = 50,
                 embedding_model: str = "shibing624/text2vec-base-chinese",
                 rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                 collection_name: str = "rag_documents",
                 log_file: str = "rag_system.log"):
        """
        初始化RAG系统
        
        Args:
            chunk_size: 文档分块大小
            overlap: 分块重叠大小
            embedding_model: 向量化模型名称
            rerank_model: 重排序模型名称
            collection_name: ChromaDB集合名称
            log_file: 日志文件路径
        """
        # 初始化日志系统
        self.logger = RAGLogger(log_file)
        self.logger.info("🚀 RAG系统启动中...")
        
        # 初始化各个组件
        self.doc_processor = DocumentProcessor(chunk_size, overlap, self.logger)
        self.embedder = VectorEmbedder(embedding_model, self.logger)
        self.db_manager = ChromaDBManager(collection_name, self.logger)
        self.reranker = ResultReranker(rerank_model, self.logger)
        self.generator = DeepSeekGenerator(self.logger)
        
        self.logger.success("RAG系统初始化完成")
    
    def add_document(self, text: str) -> int:
        """
        添加文档到系统
        
        Args:
            text: 文档内容
            
        Returns:
            添加的文档块数量
        """
        try:
            self.logger.info("开始处理新文档")
            
            # 1. 分割文档
            chunks = self.doc_processor.split_text(text)
            
            if not chunks:
                self.logger.warning("文档分割后为空")
                return 0
            
            # 2. 向量化
            embeddings = self.embedder.embed_batch(chunks)
            
            # 3. 存储到数据库
            self.db_manager.add_documents(chunks, embeddings)
            
            self.logger.success(f"文档添加完成 - 共 {len(chunks)} 个块")
            return len(chunks)
            
        except Exception as e:
            self.logger.error(f"文档添加失败: {e}")
            raise
    
    def add_document_from_file(self, file_path: str) -> int:
        """
        从文件添加文档
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            添加的文档块数量
        """
        try:
            # 加载文档
            content = self.doc_processor.load_document(file_path)
            
            # 添加到系统
            return self.add_document(content)
            
        except Exception as e:
            self.logger.error(f"从文件添加文档失败: {e}")
            raise
    
    def query(self, question: str, top_k: int = 5, rerank_top_k: int = 3) -> dict:
        """
        执行RAG查询
        
        Args:
            question: 用户问题
            top_k: 初始检索的文档数量
            rerank_top_k: 重排序后保留的文档数量
            
        Returns:
            包含答案和相关信息的字典
        """
        try:
            self.logger.info(f"开始处理查询: {question}")
            start_time = time.time()
            
            # 1. 向量化查询
            query_embedding = self.embedder.embed_text(question)
            
            # 2. 检索相关文档
            documents, distances = self.db_manager.search(query_embedding, top_k)
            
            if not documents:
                self.logger.warning("未找到相关文档")
                return {
                    "answer": "抱歉，我没有找到相关的信息来回答您的问题。",
                    "documents": [],
                    "query_time": time.time() - start_time
                }
            
            # 3. 重排序
            reranked_docs = self.reranker.rerank(question, documents, rerank_top_k)
            
            # 4. 生成答案
            answer = self.generator.generate_answer(question, reranked_docs)
            
            query_time = time.time() - start_time
            self.logger.success(f"查询完成 - 耗时: {query_time:.2f}秒")
            
            return {
                "answer": answer,
                "documents": reranked_docs,
                "query_time": query_time,
                "retrieved_count": len(documents),
                "reranked_count": len(reranked_docs)
            }
            
        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            raise
    
    def get_system_status(self) -> dict:
        """
        获取系统状态信息
        
        Returns:
            系统状态字典
        """
        try:
            db_info = self.db_manager.get_collection_info()
            
            return {
                "status": "running",
                "database": db_info,
                "components": {
                    "document_processor": "active",
                    "embedder": "active" if self.embedder.model else "inactive",
                    "database": "active" if self.db_manager.collection else "inactive",
                    "reranker": "active" if self.reranker.model else "inactive",
                    "generator": "active" if self.generator.client else "inactive"
                }
            }
            
        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return {"status": "error", "error": str(e)}


def main():
    """
    主函数 - 演示RAG系统的使用
    """
    try:
        # 创建RAG系统实例
        rag = RAGSystem()
        
        # 示例文档内容（来自notebook的故事）
        sample_document = """
        在一个平凡的下午，哆啦A梦正在大雄房间里整理道具，突然一道蓝色光芒闪过，一个紫发少年从时空裂缝中跌落出来。这个少年正是来自未来的特兰克斯，他神情严肃，眼中带着深深的忧虑。
        
        "哆啦A梦，大雄，"特兰克斯喘着粗气说道，"我需要你们的帮助。未来世界正面临前所未有的危机，只有你们的科技力量才能拯救一切。"
        
        哆啦A梦与大雄听后大惊，但也从特兰克斯坚定的眼神中读出了不容拒绝的决心。特兰克斯解释说，未来的敌人并非普通反派，而是一个名叫"黑暗赛亚人"的存在，他由邪恶科学家复制了贝吉塔的基因并加以改造，实力超乎想象。这个敌人不仅拥有赛亚人战斗力，还能操纵扭曲的时间能量，几乎无人可敌。特兰克斯已经独自战斗多年，但每一次都以惨败告终。他说："科技，是我那个时代唯一缺失的武器，而你们，正好拥有它。"
        
        哆啦A梦听完后沉思片刻，然后从四次元口袋中取出了三件秘密道具。三件秘密道具分别是：可以临时赋予超级战力的"复制斗篷"，能暂停时间五秒的"时间停止手表"，以及可在一分钟中完成一年修行的"精神与时光屋便携版"。大雄被推进精神屋内，在其中接受密集的训练，虽然只有几分钟现实时间，他却经历了整整一年的苦修。刚开始他依旧软弱，想放弃、想逃跑，但当他想起静香、父母，还有哆啦A梦那坚定的眼神时，他终于咬牙坚持了下来。出来之后，他的身体与精神都焕然一新，眼神中多了一份成熟与自信。
        
        最终战在黑暗赛亚人的空中要塞前爆发，特兰克斯率先出击，释放全力与敌人正面对决。哆啦A梦则用任意门和道具支援，从各个方向制造混乱，尽量压制敌人的时空能力。但黑暗赛亚人太过强大，仅凭特兰克斯一人根本无法压制，更别说击败。就在特兰克斯即将被击倒之际，大雄披上复制斗篷、冲破恐惧从高空跃下。他的拳头燃烧着金色光焰，目标直指敌人心脏。
        
        战后，未来世界开始恢复，植物重新生长，人类重建家园。特兰克斯告别时紧紧握住大雄的手，说："你是我见过最特别的战士。"哆啦A梦也为大雄感到骄傲，说他终于真正成长了一次。三人站在山丘上，看着远方重新明亮的地平线，心中感受到从未有过的安宁。随后，哆啦A梦与大雄乘坐时光机返回了属于他们的那个年代，一切仿佛又恢复平静。
        """
        
        # 添加示例文档
        print("\n📚 添加示例文档...")
        chunk_count = rag.add_document(sample_document)
        print(f"✅ 文档处理完成，共生成 {chunk_count} 个文本块")
        
        # 显示系统状态
        print("\n📊 系统状态:")
        status = rag.get_system_status()
        print(f"状态: {status['status']}")
        print(f"文档数量: {status['database']['document_count']}")
        
        # 执行示例查询
        print("\n🔍 执行示例查询...")
        questions = [
            "哆啦A梦使用的3个秘密道具分别是什么？",
            "大雄在精神与时光屋中训练了多长时间？",
            "特兰克斯来自哪里，他遇到了什么问题？"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- 查询 {i} ---")
            print(f"问题: {question}")
            
            result = rag.query(question)
            
            print(f"\n答案: {result['answer']}")
            print(f"查询耗时: {result['query_time']:.2f}秒")
            print(f"检索文档数: {result['retrieved_count']}")
            print(f"重排序后: {result['reranked_count']}")
            
            # 显示相关文档片段
            print("\n相关文档片段:")
            for j, doc in enumerate(result['documents'][:2], 1):
                print(f"{j}. {doc[:100]}...")
        
        print("\n🎉 RAG系统演示完成！")
        
    except Exception as e:
        print(f"❌ 系统运行出错: {e}")
        raise


if __name__ == "__main__":
    main()