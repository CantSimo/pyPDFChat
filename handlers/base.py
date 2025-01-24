import tempfile
import os
import tiktoken

from utils.alerts import alert_exception, alert_info
from fastapi import UploadFile
from fastapi import HTTPException
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import (create_history_aware_retriever, create_retrieval_chain)
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import (
    TokenTextSplitter,
    TextSplitter,
    Tokenizer,
    Language,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    LatexTextSplitter,
    PythonCodeTextSplitter,
    KonlpyTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    SentenceTransformersTokenTextSplitter,
    ElementType,
    HeaderType,
    LineType,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    CharacterTextSplitter,
)

load_dotenv()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

class BaseHandler():
    def __init__(
            self,
            chat_model: str = 'gpt-4o-mini', 
            temperature: float = 0.2,
            **kwargs
        ):

        self.CHROMA_PERSIST_PATH = os.getenv('CHROMA_PERSIST_PATH')
        self.DOCS_TMP_PATH = os.getenv('DOCS_TMP_PATH')

        self.chat_model = chat_model
        self.llm_map = {
            'gpt-4o-mini': lambda _: ChatOpenAI(model='gpt-4o-mini', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-4o': lambda _: ChatOpenAI(model='gpt-4', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
        }

        if kwargs.get('embeddings_model') == 'text-embedding-3-large':
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv('OPENAI_API_KEY'), 
                model='text-embedding-3-large'
            )
            self.dimensions = 3072
        else:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv('OPENAI_API_KEY'), 
                model='text-embedding-3-small'
            )
            self.dimensions = 1536

    def load_documents(self, files: list[UploadFile]) -> list[list[str]]:
        documents = []

        loader_map = {
            'txt': TextLoader,
            'pdf': PyMuPDFLoader, 
            'docx': Docx2txtLoader,
        }

        allowed_extensions = [key for key in loader_map.keys()]
        try: 
            for file in files:
                if file.filename.split(".")[-1] not in allowed_extensions:
                    raise HTTPException(status_code=400, detail="File type not permitted")
                
                dir_path = self.DOCS_TMP_PATH

                with tempfile.NamedTemporaryFile(delete=False, prefix=file.filename + '___', dir=dir_path) as temp:
                    temp.write(file.file.read())
                    temp.seek(0)
                    loader = loader_map[file.filename.split(".")[-1]](temp.name)
                    documents.append(loader.load())
                    temp.close()

        except Exception as e:
            alert_exception(e, "Error loading documents")
            raise HTTPException(status_code=500, detail=f"Error loading documents: {str(e)}")
       
        return documents

    def ingest_documents(self, 
                         documents: list[list[str]], 
                         chunk_size: int = 1000, 
                         chunk_overlap: int = 200, 
                         **kwargs):
        """
        Suddivide i documenti e li indicizza in ChromaDB locale.

        Parametri:
            documents: lista di documenti caricati
            chunk_size: numero massimo di caratteri per chunk
            chunk_overlap: sovrapposizione tra i chunk
            split_method: metodo di splitting (es. 'recursive', 'token' etc.)
            namespace: opzione per gestire diversi 'collection_name'
        """
        
        splitter_map = {
            'recursive': RecursiveCharacterTextSplitter,
            'token': TokenTextSplitter,
            'text': TextSplitter,
            'tokenizer': Tokenizer,
            'language': Language,
            'json': RecursiveJsonSplitter,
            'latex': LatexTextSplitter,
            'python': PythonCodeTextSplitter,
            'konlpy': KonlpyTextSplitter,
            'spacy': SpacyTextSplitter,
            'nltk': NLTKTextSplitter,
            'sentence_transformers': SentenceTransformersTokenTextSplitter,
            'element_type': ElementType,
            'header_type': HeaderType,
            'line_type': LineType,
            'html_header': HTMLHeaderTextSplitter,
            'markdown_header': MarkdownHeaderTextSplitter,
            'markdown': MarkdownTextSplitter,
            'character': CharacterTextSplitter
        }

        split_method = kwargs.get('split_method', 'recursive')
        test_splitter = splitter_map[split_method](chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        alert_info(f"Ingesting {len(documents)} document(s)...\nParams: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, split_method={split_method}")

        persist_directory = self.CHROMA_PERSIST_PATH
        collection_name = kwargs.get('namespace', 'default_collection')

        # 1) Carica (o crea se non esiste) la collezione
        vectorstore = Chroma(
            embedding_function=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        for document in documents:
            split_document = test_splitter.split_documents(document)  

            tokens = 0
            for chunk in split_document:
                tok = num_tokens_from_string(chunk.page_content, "cl100k_base")
                # alert_info(f"curr chunk token: {tok}")
                tokens = tokens + tok
            
            alert_info(f"******** current document total token ********: {tokens}")

            try:
                # 2) Aggiungi i chunk alla collezione esistente
                vectorstore.add_documents(split_document)
            except Exception as e:
                alert_exception(e, "Error ingesting documents - Make sure you\'re dimensions match the embeddings model (1536 for text-embedding-3-small, 3072 for text-embedding-3-large)")
                raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")
                
    def chat(self, query: str, chat_history: list[str] = [], **kwargs):
        """
        Esegue una chat query utilizzando i documenti indicizzati in ChromaDB.

        Parametri:
            query: la query dell'utente
            chat_history: cronologia chat
            namespace: corrisponde a 'collection_name' in Chroma
            search_kwargs: parametri per la ricerca (es. {'k': 5})
        """      

        # alert_info(f"Querying with: {query} and chat history: {chat_history}\nParams: namespace={kwargs.get('namespace', None)}, search_kwargs={kwargs.get('search_kwargs', {'k': 5})}\nModel: {self.llm.model_name} with temperature: {self.llm.temperature}")
        try: 
            collection_name = kwargs.get('namespace', 'default_collection')

            vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.CHROMA_PERSIST_PATH
            )

            retriever = vectorstore.as_retriever(search_kwargs=kwargs.get('search_kwargs', {"k": 5}))
            
            # 1 - Prompt per riformulare la domanda rispetto alla chat history
            contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""

            prompt_search_query = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user","{input}"),
            ])           

            # 2 - Usare create_history_aware_retriever
            llm=self.llm_map[self.chat_model] 
            retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)   

            # 3 - Prompt per ottenere la risposta
            prefixSystem = "You are an expert Italian accountant, the topic is the compilation of the PF form of the Italian state."
            prompt_get_answer = ChatPromptTemplate.from_messages([
                ("system", prefixSystem),
                ("system", "Answer the user's questions based on the below context, if the context doesn't contain any relevant information to the question, don't make something up and just say 'I don't know':\\n\\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user","{input}"),
            ])   

            # 4 Document Chain
            #  we create a chain using create_stuff_documents_chain which will send the prompt to the llm.
            document_chain=create_stuff_documents_chain(llm,prompt_get_answer)

            #5 Conversational Retrieval Chain 
            # So, in the final step, we combine retriever_chain and document_chain using create_retrieval_chain to create a Conversational retrieval chain.
            retrieval_chain = create_retrieval_chain(retriever_chain, document_chain) 

            with get_openai_callback() as cb:
                response = retrieval_chain.invoke({
                "chat_history":chat_history,
                "input":query
                })

            # Extract relevant information from cb
            openai_fee = {
                "completion_tokens": cb.completion_tokens,  
                "prompt_tokens": cb.prompt_tokens ,
                "total_cost": cb.total_cost
            }

            return {
                "model_response": response,
                "model_fee": openai_fee
            }
            
        except Exception as e:
            alert_exception(e, "Error chatting")
            raise HTTPException(status_code=500, detail=f"Error chatting: {str(e)}")
