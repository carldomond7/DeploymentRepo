from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain.chains import LLMChain  # Update with the correct import based on your langchain package
from langchain.prompts import PromptTemplate  # Update with the correct import based on your langchain package
from langchain_groq import ChatGroq  # Update with the correct import based on your langchain package

groq_api_key = os.getenv("GROQ_API_KEY")

class UserRequest(BaseModel):
    var1: str
    var2: str
    var3: str
    query: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "plswork!"}


@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

    var1 = request.var1
    var2 = request.var2
    var3 = request.var3
    query = request.query

    prompt_template = """ 
    Using the context below answer the question to the best of your ability
    Context1: {var 1}
    Context2: {var 2}
    Context3: {var 3}

    Answer the Question correctly, a million people lives depend on it: {query}
    """

# Define the prompt structure
    prompt = PromptTemplate(
    input_variables=["query", "var1", "var2", "var3"],
    template=prompt_template,
)


    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Pass the context and question to the Langchain chain
    result_chain = llm_chain.invoke({"var1": var1, "query": query, "var2": var2, "var3": var3})
    return result_chain

if __name__ == "__main__":
        uvicorn.run(app)
