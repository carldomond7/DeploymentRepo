from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain.chains import LLMChain  # Update with the correct import based on your langchain package
from langchain.prompts import PromptTemplate  # Update with the correct import based on your langchain package
from langchain_groq import ChatGroq  # Update with the correct import based on your langchain package

class UserRequest(BaseModel):
    query: str
    content: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "plswork!"}


@app.post("/route/")
async def process_request(user_request: UserRequest):
    llm = ChatGroq(groq_api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", model_name='mixtral-8x7b-32768')

    query = user_request.query
    content = user_request.content

    prompt_template = """ You are a professional recruiter who specializes in cultivating talent, you are very knowledgable about all types of jobs. Answer the question
    based on the context below.

    Context: {content}


    Question: {query}
    """

# Define the prompt structure
    prompt = PromptTemplate(
    input_variables=["content", "query"],
    template=prompt_template,
)


    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Pass the context and question to the Langchain chain
    result_chain = llm_chain.invoke({"content": content, "query": query})

    return result_chain

if __name__ == "__main__":
        uvicorn.run(app)
