from prompt_templates import PromptTemplate
from process_summaries import Summary   
from llm import LLM
from langchain.chains.retrieval_qa.base import RetrievalQA


# model
llm = LLM()


# inputs
input_book_name = input("Enter the book name: ").strip().lower()
input_question = input("Enter the question: ")

summaries = Summary()
summaries.process_summaries(r"C:\Users\yunus\OneDrive\Masaüstü\Book Assistant\Summaries")

vector_store = summaries.vector_store[input_book_name]
retriever = vector_store.as_retriever()

rag_chain = RetrievalQA.from_chain_type(llm=llm.model, retriever=retriever)
response = rag_chain.invoke({"query": input_question})
print(f"\nAnswer from LLM: {response['result']}")


# llm prompt
# prompt_template = PromptTemplate(input_book_name, input_question)
# final_prompt = prompt_template.get_prompt()

# prompt -> model
# response = llm.model.invoke(final_prompt)
# print(f"\nAnswer from LLM: {response}")
