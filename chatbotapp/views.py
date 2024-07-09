from django.shortcuts import render, HttpResponse
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from django.conf import settings
api_key = settings.API_KEY
# # from chatbotapp.models import Company, Employee
# # from chatbotapp.serializers import CompanySerializer, EmployeeSerializer
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.authentication import BasicAuthentication
from rest_framework.permissions import IsAuthenticated
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain import HuggingFaceHub
from django.http import StreamingHttpResponse
from asgiref.sync import async_to_sync


embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
)

pdf_files = ["sample2.pdf", "sample1.pdf"]  # Add all your PDF filenames here

# Load and combine documents from all PDFs
all_pages = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file, extract_images=True)
    pages = loader.load()
    all_pages.extend(pages)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(all_pages)

print(docs)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()


# API by decorators
@api_view(["POST"])
def RagViewSet(request):
    try:
        question = request.data.get("question", None)
        if question is not None:
            print("Rag api calling", question)

            template = """Greet if anyone say hi or hello or how are you. You are a helping chatbot for a tech website and your job is to facilitate people
        with their questions. Be polite and Answer the question based on the context provided.
        Use the word our not their and don't mention anything about the documents provided for context
        Don't reply with Based on the context provided just make it concise and accurate.
        And don't cut off any sencence from last.
        Answer should never be more than 50 words i repeat never
        Don't say The document does not provide a specific.
        If you can't answer the question, just say I can connect you with our team who can assist you further".
        Question: {question}
        Context: {context}
        Answer:
        """

            prompt = ChatPromptTemplate.from_template(template)

            output_parser = StrOutputParser()

            model = HuggingFaceHub(
                huggingfacehub_api_token=api_key,
                repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                model_kwargs={"temperature": 1, "max_length": 3500},
            )

            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | output_parser
            )

            answer = rag_chain.invoke(question)
            print(rag_chain, answer, "here")
            split_text = answer.split("Answer:")
            final = split_text[1].strip()
            print(final)

            return Response(
                {
                    "message": "Request received successfully",
                    "question": question,
                    "answer": final,
                }
            )
        else:
            return Response(
                {"error": "Invalid request data. 'question' key is missing."},
                status=400,
            )
    except Exception as e:
        return Response({"error": str(e)}, status=400)


# # import pickle
# # import faiss
# # from django.core.cache import cache
# # from rest_framework.decorators import api_view
# # from langchain.vectorstores import FAISS
# # from langchain.prompts import ChatPromptTemplate
# # from langchain.schema.output_parser import StrOutputParser
# # from langchain import HuggingFaceHub
# # from .serializers import UserSerializer
# # from .models import User
# # import jwt, datetime
# # import os


# # Create your views here.
# def appview(request):
#     return HttpResponse("This is App page")


# # class CompanyViewSet(viewsets.ModelViewSet):
# #     queryset = Company.objects.all()
# #     serializer_class = CompanySerializer

# #     @action(detail=True, methods=["get"])
# #     def employees(self, request, pk=None):
# #         try:
# #             company = Company.objects.get(pk=pk)
# #             emps = Employee.objects.filter(company=company)
# #             emps_serializer = EmployeeSerializer(
# #                 emps, many=True, context={"request": request}
# #             )
# #             return Response(emps_serializer.data)
# #         except Exception as e:
# #             print(e)
# #             return Response({"message": "Company might not exists !! Error"})


# # class EmployeeViewSet(viewsets.ModelViewSet):
# #     queryset = Employee.objects.all()
# #     serializer_class = EmployeeSerializer
# #     authentication_classes = [BasicAuthentication]
# #     permission_classes = [IsAuthenticated]


# # the api can be made by decorators or by making a function
# # API by function
# # Create your views here.
# # class RegisterView(APIView):
# #   def post(self, request):
# #     serializer = UserSerializer(data=request.data)
# #     serializer.is_valid(raise_exception=True)
# #     serializer.save()
# #     return Response(serializer.data)

# # #API by decorators
# # @api_view(['POST'])
# # def LoginView(request):
# #   email = request.data['email']
# #   password = request.data['password']

# #   user = User.objects.filter(email=email).first()

# #   if user is None:
# #     raise AuthenticationFailed('User not found!')

# #   if not user.check_password(password):
# #     raise AuthenticationFailed('Incorrect password!')

# #   payload = {
# #       'id': user.id,
# #       'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
# #       'iat': datetime.datetime.utcnow()
# #   }

# #   token = jwt.encode(payload, 'secret', algorithm='HS256')

# #   response = Response()

# #   response.set_cookie(key='jwt', value=token, httponly=True)
# #   response.data = {
# #       'jwt': token
# #   }
# #   return response


# # # class UserView(APIView):
# # @api_view(['GET'])
# # def UserView(request):
# #   token = request.COOKIES.get('jwt')

# #   if not token:
# #     raise AuthenticationFailed('Unauthenticated!')

# #   try:
# #     payload = jwt.decode(token, 'secret', algorithms=['HS256'])
# #   except jwt.ExpiredSignatureError:
# #       raise AuthenticationFailed('Unauthenticated!')

# #   user = User.objects.filter(id=payload['id']).first()
# #   serializer = UserSerializer(user)
# #   return Response(serializer.data)


# # # class LogoutView(APIView):
# # @api_view(['POST'])
# # def LogoutView(request):
# # response = Response()
# # response.delete_cookie('jwt')
# # response.data = {
# #     'message': 'success'
# # }
# # return response

# specify embedding model (using huggingface sentence transformer)


# Load vectorstore once and cache it
# Load FAISS index and document store
# def load_vectorstore():
#     index = faiss.read_index("faiss_index.index")
#     with open("vectorstore.pkl", "rb") as f:
#         vectorstore = pickle.load(f)
#     vectorstore.index = index
#     return vectorstore

# vectorstore = load_vectorstore()

# model = HuggingFaceHub(
#     huggingfacehub_api_token='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',
#     repo_id="mistralai/Mistral-7B-Instruct-v0.1",
#     model_kwargs={"temperature": 1, "max_length": 180}
# )

# template = """
# You are a helping chatbot for a tech website and your job is to facilitate people
# with their questions. Be polite and Answer the question based on the context provided.
# Use the word our not their and don't mention anything about the documents provided for context
# Don't reply with based on the context below just make it concise and accurate.
# Don't say The document does not provide a specific
# If you can't answer the question, just say I can connect you with our team who can assist you further".
# Question: {question}
# Context: {context}
# Answer:
# """
# prompt = ChatPromptTemplate.from_template(template)
# output_parser = StrOutputParser()

# @api_view(['POST'])
# def RagViewSet(request):
#   question = request.data.get('question', None)
#   if question is None:
#     return Response({"error": "Invalid request data. 'question' key is missing."}, status=400)

#   cache_key = f"answer_{hash(question)}"
#   answer = cache.get(cache_key)
#   if answer is not None:
#     return Response({"message": "Request received successfully", "question": question, "answer": answer})

#   try:
#     retriever = vectorstore.as_retriever()
#     rag_chain = (
#       {"context": retriever, "question": question}
#       | prompt
#       | model
#       | output_parser
#     )

#     answer = rag_chain.invoke(question)
#     split_text = answer.split("Answer:")
#     final_answer = split_text[1].strip() if len(split_text) > 1 else answer.strip()

#     # Cache the result
#     cache.set(cache_key, final_answer, timeout=60*60)  # cache for 1 hour

#     return Response({"message": "Request received successfully", "question": question, "answer": final_answer})
#   except Exception as e:
#     return Response({"error": str(e)}, status=400)

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.prompts import ChatPromptTemplate
# from langchain_huggingface import HuggingFacePipeline
# from langchain_community.vectorstores import FAISS
# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import PromptTemplate

# # Specify embedding model (using huggingface sentence transformer)
# embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
# embeddings = HuggingFaceEmbeddings(
#     model_name=embedding_model_name,
# )

# loader = PyPDFLoader("sample.pdf", extract_images=True)
# pages = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# docs = text_splitter.split_documents(pages)

# vectorstore = FAISS.from_documents(docs, embeddings)
# retriever = vectorstore.as_retriever()

# # Load the flan-t5 model and tokenizer
# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
# nlp_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# # Define the prompt template
# template = """You are a helpful chatbot for a tech website and your job is to assist people
# with their questions. Be polite and answer the question based on the context provided.
# Use the word 'our' not 'their' and don't mention anything about the documents provided for context.
# Don't reply with "Based on the context provided", just make it concise and accurate.
# Answer the question fully and do not cut off any sentence from last.
# Don't say "The document does not provide a specific answer."
# If you can't answer the question, just say "I can connect you with our team who can assist you further."
# Question: {question}
# Context: {context}
# Answer:
# """
# prompt_template = PromptTemplate(input_variables=["question", "context"], template=template)

# def truncate_context_to_fit(question, context, tokenizer, max_length=512):
#     # Tokenize the question and context
#     question_tokens = tokenizer.encode(question, add_special_tokens=False)
#     context_tokens = tokenizer.encode(context, add_special_tokens=False)

#     # Calculate the available length for the context
#     available_length = max_length - len(question_tokens) - 10  # Reserve some space for the prompt and answer

#     # Truncate the context tokens if necessary
#     if len(context_tokens) > available_length:
#         context_tokens = context_tokens[:available_length]

#     # Decode the truncated context tokens back to text
#     truncated_context = tokenizer.decode(context_tokens, skip_special_tokens=True)
#     return truncated_context

# # API by decorators
# @api_view(["POST"])
# def RagViewSet(request):
#     try:
#         question = request.data.get("question", None)
#         if question is not None:
#             print("Rag api calling", question)

#             # Retrieve relevant documents
#             context_docs = retriever.get_relevant_documents(question)
#             context = " ".join([doc.page_content for doc in context_docs])
#             print("Retrieved Context:", context)  # Debug print to see the retrieved context

#             # Truncate context to fit within the token limit
#             context = truncate_context_to_fit(question, context, tokenizer)
#             print("Truncated Context:", context)  # Debug print to see the truncated context

#             prompt = prompt_template.format(question=question, context=context)
#             print("Generated Prompt:", prompt)  # Debug print to see the prompt

#             # Generate the response using the flan-t5 pipeline
#             response = nlp_pipeline(prompt, max_new_tokens=1000, num_return_sequences=1, clean_up_tokenization_spaces=True)
#             answer = response[0]['generated_text'].strip()

#             print("Rag API response:", answer)

#             return Response(
#                 {
#                     "message": "Request received successfully",
#                     "question": question,
#                     "answer": answer,
#                 }
#             )
#         else:
#             return Response(
#                 {"error": "Invalid request data. 'question' key is missing."},
#                 status=400,
#             )
#     except Exception as e:
#         return Response({"error": str(e)}, status=400)

# import asyncio
# from django.http import StreamingHttpResponse
# from rest_framework.decorators import api_view
# from rest_framework.response import Response

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# from langchain_community.llms import HuggingFaceHub  # Updated import

# embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
# embeddings = HuggingFaceEmbeddings(
#     model_name=embedding_model_name,
# )

# pdf_files = ["sample2.pdf", "sample1.pdf"]  # Add all your PDF filenames here

# # Load and combine documents from all PDFs
# all_pages = []
# for pdf_file in pdf_files:
#     loader = PyPDFLoader(pdf_file, extract_images=True)
#     pages = loader.load()
#     all_pages.extend(pages)

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# docs = text_splitter.split_documents(all_pages)

# print(docs)

# vectorstore = FAISS.from_documents(docs, embeddings)
# retriever = vectorstore.as_retriever()

# template = """Greet if anyone say hi or hello or how are you. You are a helping chatbot for a tech website and your job is to facilitate people
# with their questions. Be polite and Answer the question based on the context provided.
# Use the word our not their and don't mention anything about the documents provided for context
# Don't reply with Based on the context provided just make it concise and accurate.
# And don't cut off any sencence from last.
# Answer should never be more than 50 words i repeat never
# Don't say The document does not provide a specific.
# If you can't answer the question, just say I can connect you with our team who can assist you further".
# Question: {question}
# Context: {context}
# Answer:
# """

# prompt = ChatPromptTemplate.from_template(template)
# output_parser = StrOutputParser()

# model = HuggingFaceHub(
#     huggingfacehub_api_token=process.env.API_KEY,
#     repo_id="mistralai/Mistral-7B-Instruct-v0.1",
#     model_kwargs={"temperature": 1, "max_length": 3500},
# )

# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | model
#     | output_parser
# )

# async def generate_response(question):
#     try:
#         print(f"Invoking model with question: {question}")
#         answer = rag_chain.invoke(question)
#         print(f"Raw answer: {answer}")
#         split_text = answer.split("Answer:")
#         final = split_text[1].strip() if len(split_text) > 1 else answer.strip()
#         return f"{final}\n"
#     except Exception as e:
#         print(f"Error in generate_response: {str(e)}")
#         return f"Error: {str(e)}"

# @api_view(["POST"])
# async def RagViewSet(request):
#     try:
#         question = request.data.get("question", None)
#         if question is not None:
#             print("Rag api calling", question)

#             async def stream_response():
#                 try:
#                     yield await generate_response(question)
#                 except Exception as e:
#                     yield f"Error: {str(e)}"

#             return StreamingHttpResponse(
#                 stream_response(),
#                 content_type='text/plain'
#             )
#         else:
#             return Response(
#                 {"error": "Invalid request data. 'question' key is missing."},
#                 status=400,
#             )
#     except Exception as e:
#         print(f"Error in RagViewSet: {str(e)}")
#         return Response({"error": str(e)}, status=400)

