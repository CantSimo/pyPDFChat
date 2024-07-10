import tempfile
import os
from utils.alerts import alert_exception, alert_info
from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain.chains import (LLMChain, ConversationalRetrievalChain)
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from fastapi import UploadFile
from fastapi import HTTPException
from dotenv import load_dotenv
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

class BaseHandler():
    def __init__(
            self,
            chat_model: str = 'gpt-3.5-turbo',
            temperature: float = 0.7,
            **kwargs
        ):

        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
        self.pinecone_index = os.getenv('PINECONE_INDEX')
        self.llm_map = {
            'gpt-4': lambda: ChatOpenAI(model='gpt-4', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-4-32k': lambda: ChatOpenAI(model='gpt-4-32k', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-4-1106-preview': lambda: ChatOpenAI(model='gpt-4', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-3.5-turbo-16k': lambda: ChatOpenAI(model='gpt-3.5-turbo-16k', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-3.5-turbo': lambda: ChatOpenAI(model='gpt-3.5-turbo', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'claude-3-sonnet-20240229': lambda: ChatAnthropic(model_name='claude-3-sonnet-20240229', temperature=temperature, anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')),
            'claude-3-opus-20240229': lambda: ChatAnthropic(model_name='claude-3-opus-20240229', temperature=temperature, anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')),
        }
        self.chat_model = chat_model
        # self.streaming_llm = ChatOpenAI(
        #     model=openai_chat_model,
        #     streaming=True,
        #     callbacks=[StreamingStdOutCallbackHandler()],
        #     temperature=0,
        #     openai_api_key=os.getenv('OPENAI_API_KEY'),
        # )

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

    def load_documents(self, files: list[UploadFile], namespace: str = None) -> list[list[str]]:
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
                
                # Get the directory of the original file
                # TODO:: Parametrizzare
                dir_path = "D:\\MyApps\\pyPDFChat\\docs"

                with tempfile.NamedTemporaryFile(delete=False, prefix=file.filename + '___', dir=dir_path) as temp:
                    temp.write(file.file.read())
                    temp.seek(0)
                    loader = loader_map[file.filename.split(".")[-1]](temp.name)
                    documents.append(loader.load())

        except Exception as e:
            alert_exception(e, "Error loading documents")
            raise HTTPException(status_code=500, detail=f"Error loading documents: {str(e)}")

        
        return documents

    def ingest_documents(self, documents: list[list[str]], chunk_size: int = 1000, chunk_overlap: int = 100, **kwargs):
        """
        documents: list of loaded documents
        chunk_size: number of documents to ingest at a time
        chunk_overlap: number of documents to overlap when ingesting

        kwargs:
            split_method: 'recursive', 'token', 'text', 'tokenizer', 'language', 'json', 'latex', 'python', 'konlpy', 'spacy', 'nltk', 'sentence_transformers', 'element_type', 'header_type', 'line_type', 'html_header', 'markdown_header', 'markdown', 'character'
        """
        pc = Pinecone(api_key = self.pinecone_api_key)
        pc.list_indexes().names()

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
        for document in documents:
            split_document = test_splitter.split_documents(document)  
            try:
                PineconeVectorStore.from_documents(
                    split_document, 
                    self.embeddings, 
                    index_name=self.pinecone_index, 
                    namespace=kwargs.get('namespace', None) # You can only specify a namespace if you have a premium Pinecone pod
                )
            except Exception as e:
                alert_exception(e, "Error ingesting documents - Make sure you\'re dimensions match the embeddings model (1536 for text-embedding-3-small, 3072 for text-embedding-3-large)")
                raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")
                
    def chat(self, query: str, chat_history: list[str] = [], **kwargs):
        """
        query: str
        chat_history: list of previous chat messages
        kwargs:
            namespace: str
            search_kwargs: dict
        """

        # OK FUNZIONA
        # ChatGPT bot test
        # bot = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7, openai_api_key=os.getenv('OPENAI_API_KEY'))
        # bot.invoke("Ciao, dimmmi qualcosa")

        # alert_info(f"Querying with: {query} and chat history: {chat_history}\nParams: namespace={kwargs.get('namespace', None)}, search_kwargs={kwargs.get('search_kwargs', {'k': 5})}\nModel: {self.llm.model_name} with temperature: {self.llm.temperature}")
        try: 
            pc = Pinecone(api_key = self.pinecone_api_key)

            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=self.pinecone_index, 
                embedding=self.embeddings, 
                text_key='text', 
                namespace=kwargs.get('namespace', None) # You can only specify a namespace if you have a premium Pinecone pod
            )

            retriever = vectorstore.as_retriever(search_kwargs=kwargs.get('search_kwargs', {"k": 5}))


            # FAIL
            # bot = ConversationalRetrievalChain.from_llm(
            #     self.llm_map[self.chat_model], 
            #     retriever, 
            #     return_source_documents=True
            # )
            # result = bot.invoke({"question": query, "chat_history": chat_history})

            # FAIL
            # template = (
            #     "Combine the chat history and follow up question into "
            #     "a standalone question. Chat History: {chat_history}"
            #     "Follow up question: {query}"
            # )
            # prompt = PromptTemplate.from_template(template)
            # question_generator_chain = LLMChain(llm=self.llm_map[self.chat_model], prompt=prompt)
            # result = ConversationalRetrievalChain(retriever=retriever,
            #     question_generator=question_generator_chain,
            # )
 
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm_map[self.chat_model],
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={
                        "prompt": ChatPromptTemplate.from_messages(
                            [
                                "",
                                query,
                            ]
                        ),
                    },
                )
            
            return conversation_chain
        except Exception as e:
            alert_exception(e, "Error chatting")
            raise HTTPException(status_code=500, detail=f"Error chatting: {str(e)}")
        # for chunk in result:
        #     yield chunk