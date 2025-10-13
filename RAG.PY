# 导入相关模块和类
from langchain_community.retrievers import BM25Retriever
from typing import List
import jieba
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

# 加载文本数据并分割
loader = TextLoader("medical_data.txt")  # 加载文本文件
documents = loader.load()

# 使用递归字符分割器进行文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,             # 每块最多500个字符
    chunk_overlap=0,            # 不重叠
    length_function=len,
    separators=['\n']           # 按换行符切分
)
docs = text_splitter.split_documents(documents)  # 分割后的文档

# 定义中文分词的预处理函数
def preprocessing_func(text: str) -> List[str]:
    return list(jieba.cut(text))  # 使用jieba进行中文分词

# 构建BM25检索器
bm25 = BM25Retriever(docs=docs, k=10)
print(bm25.k)
retriever = bm25.from_documents(docs, preprocess_func=preprocessing_func)

# 使用BM25检索器进行查询
retriever.invoke("胃肠功能紊乱怎么办")

# 使用自定义BM25向量模块进行检索排序
from rank_bm25 import BM25Okapi

texts = [i.page_content for i in docs]  # 提取原始文本内容
texts_processed = [preprocessing_func(t) for t in texts]  # 分词处理
vectorizer = BM25Okapi(texts_processed)  # 建立BM25索引

# 从分词后的文本中获取与查询最相关的前10条
vectorizer.get_top_n(preprocessing_func("胃肠功能紊乱怎么办"), texts, n=10)

# embeddings = HuggingFaceEmbeddings(
#     model_name="/home/york/Documents/v2",
#     model_kwargs={'device': 'cuda'}
# )

# # 2. 从文档构建 FAISS 向量数据库（假设 docs 已准备好）
# db = FAISS.from_documents(docs, embeddings)

# # db.save_local('your_save_path')

# # 4. 使用 BM25 检索 top-n 文本
# bm25_res = vectorizer.get_top_n(
#     preprocessing_func('骨折了应该怎么办'),  texts,n=10                                     
# )
# bm25_res


# # vector_res = db.similarity_search('骨折了应该怎么办', k=10)
# # vector_res

# # def rrf(vector_results: List[str], text_results: List[str], k: int=10, m: int=60):
# #         """
# #         使用RRF算法对两组检索结果进行重排序
        
# #         params:
# #         vector_results (list): 向量召回的结果列表,每个元素是专利ID
# #         text_results (list): 文本召回的结果列表,每个元素是专利ID
# #         k(int): 排序后返回前k个
# #         m (int): 超参数
        
# #         return:
# #         重排序后的结果列表,每个元素是(文档ID, 融合分数)
# #         """
        
# #         doc_scores = {}
        
# #         # 遍历两组结果,计算每个文档的融合分数
# #         for rank, doc_id in enumerate(vector_results):
# #             doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank+m)
# #         for rank, doc_id in enumerate(text_results):
# #             doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank+m)
        
# #         # 将结果按融合分数排序
# #         sorted_results = [d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]]

# #         return sorted_results


# # vector_results = [i.page_content for i in vector_res]
# # text_results = [i for i in bm25_res]
# # rrf_res = rrf(vector_results, text_results)
# # rrf_res