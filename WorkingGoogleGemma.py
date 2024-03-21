from langchain_community.document_loaders import TextLoader
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain import LLMChain, PromptTemplate
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

loader = TextLoader("data.csv")
document = loader.load()

# print(document)

import textwrap


def wrap_text(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)

# print(docs)
# print(len(docs))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# prompt = "What is the password of booker12??"

# doc = db.similarity_search(prompt)

# print(wrap_text(str(doc[0])))

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                     model_kwargs={"temperature": 0.9, "max_length": 64, "max_new_tokens": 1024})

memory = ConversationBufferMemory()
conversationBuff = ConversationChain(llm=llm, memory=memory)

# template = """ Question: {question}
# Answer: Let's think step by step.
# """

# prompt = PromptTemplate(template=template, input_variables=["question"])

chain = load_qa_chain(llm, chain_type="stuff")

# prompt = ("When was Narendra Modi born")
# prompt = ("Who was wife of Narendra Modi")
# prompt = ("Which year Narendra Modi got married")
# docsResult = db.similarity_search(prompt)
#
# print(chain.run(input_documents=docsResult, question=prompt))

# chain = LLMChain(prompt=prompt, llm = llm)


# Provide feedback to the LLM on the length of the expected answer
# prompt += "\nThe answer might be long. Please provide a partial answer and indicate if more information is available."

while True:
    prePrompt = "Understand the data and answer the following\n"
    print("Enter your query: \n")
    question = input(
        "Classify Query Column into Financial terms such as Financial Performance, "
        "Growth Strategy, Risk Management, Market Position, Regulatory Compliance and Corporate Governance with proper "
        "headers\n")

    question = prePrompt + question

    print(question)
    # question = (
    #     "Understand the data and Classify the query What is the expected p90 EBITDA 5 to 10 years down the line? in which field it will be categorized from the given category")

    docsResult = db.similarity_search(question)  # Adjust 'k' as needed
    # Print intermediate results for inspection
    print("Retrieved documents:")
    print(docsResult)

    print("***********RESULT***********")
    print(chain.run(input_documents=docsResult, question=question))
