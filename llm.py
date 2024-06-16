from ocr import process_image

from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.schema.runnable import RunnablePassthrough

import os, time
from dotenv import load_dotenv

load_dotenv()


def get_current_time():
    current_time_seconds = time.time()

    # Convert the time to a struct_time in local time
    local_time = time.localtime(current_time_seconds)

    # Format the time into a human-readable string
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    return formatted_time

print(f'Processing image.. {get_current_time()}')
ingredients = process_image()
# ingredients = 'Ingredients:Potato, Refined Palmolein Oil, BengalGram Flour (Besan), Potato Flakes, Potato Starch,Tapioca Starch, Tepary Beans (Moth Dal) Flour,Edible Common Salt, Red Chilli Powder, Black Salt,Acidity Regulator (INS 330),Black Pepper, Clove,Dried Ginger Powder, Dried Garlic Powder, CuminPowder, Bay Leaf, Nutmeg, Cinnamon, Turmeric,Mint Leaves, Natural Flavouring Substances& Nature Identical Flavouring Substances'
print(f'Image processed {get_current_time()}, below are the ingredients')
print(ingredients)

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
persist_dir = "./chroma_db"


def create_vector_db():
    print(f'Creating vector db: {get_current_time()}')
    loader = WebBaseLoader(web_paths=("https://en.m.wikipedia.org/wiki/List_of_allergens",))
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100).split_documents(docs)
    chunks = filter_complex_metadata(documents)

    vectordb = Chroma.from_documents(
        chunks,
        OllamaEmbeddings(model="llama3"),
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f'Finished creating vector db: {get_current_time()}')


def load_vector_db():
    print(f'Loading vector db: {get_current_time()}')
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=OllamaEmbeddings(model="llama3")
    )
    print(f'Finished loading vector db: {get_current_time()}')
    return vectordb


# create_vector_db()
vectordb = load_vector_db()

print(f'Creating retriever: {get_current_time()}')
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.5,
    },
)
print(f'retriever created: {get_current_time()}')

# Prompt Template
prompt = PromptTemplate.from_template(
    """
    <s> [INST] You are an expert in identifying allergic ingredients. Here's your guide {context} \n
    A list of comma separated ingredients are given to you as input, Iterate over each ingredient and think carefully if it causes a potential allergy.
    If yes, then add the ingredient to your output in the below format.
    'Ingredient name: concise and straight to the point 1 liner about the allergic reaction from this ingredient'
    If no, Dont output anything about the ingredient. \n
    You should output only about the allergic ingredients
    Remember that you should always give the same output no matter how many times I invoke you if the input is same
    [/INST] </s>
    [INST] {ingredients} [/INST]
    """
)

llm = Ollama(base_url="http://localhost:11434", model="llama3")

chain = (
    {"context": retriever, "ingredients": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(f'query invoked: {get_current_time()}')
out = chain.invoke(ingredients)
print(f'query returned: {get_current_time()}')

print(out)
