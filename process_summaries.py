from config import config

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document   

class Summary:
    def __init__(self):
        self.vector_store = {}
    
    def load_txt(self,file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The file {file_path} does not exist. Please check the path"
            )
        loader = TextLoader(file_path, encoding='utf-8')
        document = loader.load()
        #print(document)
        return document
                

    def chunking(self, document):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        texts = text_splitter.split_text(document)
        #print("Chunks:", texts)
        return texts

    def convert_chunks_to_documents(self, chunks):
        documents = [Document(page_content=chunk) for chunk in chunks]
        return documents

    def embedding(self, chunks):
        embedding = OllamaEmbeddings(model = config.EMBEDDING_MODEL)
        vectors = embedding.embed_documents(chunks)
        return vectors


    def save_vectors(self, chunks):
        vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model=config.EMBEDDING_MODEL))
        vector_store.add_documents(self.convert_chunks_to_documents(chunks))
        return vector_store


    def process_summaries(self,directory):
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                book_name = filename.replace(".txt", "").lower()
                file_path = os.path.join(directory, filename)
                document = self.load_txt(file_path)
                chunks = self.chunking(document[0].page_content)
                self.vector_store[book_name] = self.save_vectors(chunks)
                
            

#summary = Summary()
#document = summary.load_txt("C:\\Users\\yunus\\OneDrive\\Masaüstü\\Book Assistant\\Summaries\\Animal Farm.txt")

#print("Length:", len(document))
#print(type(document))

#print(type(document[0]))
#print("Metadata:", document[0].metadata)
#print("Page Content", document[0].page_content)
#print("Type: ", document[0].type)

#chunks = summary.chunking(document[0].page_content)

#print("Type:", type(chunks))
#print("Len Chunks: ", len(chunks))

#print("Deneme embedding:")
#vectors = summary.embedding(chunks)

#print(len(vectors))
#print(vectors[0])

''' 
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectors = embedding.embed_documents(["Bu bir test cümlesidir.", "İkinci parça."])
print(vectors)

print(type(vectors))
print(type(vectors[0]))
print("1--------------------",vectors[0])
print("2--------------------",vectors[1])

print(len(vectors))
'''

'''
# 1. Embedding modelini başlat
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# 2. Örnek chunk verileri
chunks = ["Bu bir test cümlesidir.", "İkinci parça."]

# 3. Chunk'ları Document nesnesine çevir
documents = [Document(page_content=chunk) for chunk in chunks]

# 4. InMemoryVectorStore oluştur
vector_store = InMemoryVectorStore(embedding=embedding_model)

# 5. Document'leri ekle (embedding işlemi burada otomatik yapılır)
vector_store.add_documents(documents)

# 6. Yüklemenin başarılı olduğunu test et
results = vector_store.similarity_search("par", k=2)

# 7. Sonuçları yazdır
for i, doc in enumerate(results, 1):
    print(f"\nSonuç {i}: {doc.page_content}")

'''

'''
for filename in os.listdir("C:\\Users\\yunus\\OneDrive\\Masaüstü\\Book Assistant\\Summaries"):
    print(filename)
'''
