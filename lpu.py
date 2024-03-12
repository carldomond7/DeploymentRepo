from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain.chains import LLMChain  # Update with the correct import based on your langchain package
from langchain.prompts import PromptTemplate  # Update with the correct import based on your langchain package
from langchain_groq import ChatGroq  # Update with the correct import based on your langchain package

groq_api_key = os.getenv("GROQ_API_KEY")

class UserRequest(BaseModel):
    Category: str
    SubCategory: str
    Title: str
    Objective: str
    Description: str
    DetailedPrompt: str
    Input1: str
    Input2: str
    Input3: str
    Input4: str
    Input5: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "plswork!"}


@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

    category = request.Category
    subcategory = request.SubCategory
    title = request.Title
    objective = request.Objective
    description = request.Description
    detailedprompt = request.DetailedPrompt
    input1 = request.Input1
    input2 = request.Input2
    input3 = request.Input3
    input4 = request.Input4
    input5 = request.Input5

    prompt_template = """ 
    Using the context below answer the question to the best of your ability
    {category}
    {subcategory}
    {title}
    {objective}
    {description}
    {input1}
    {input2}
    {input3}
    {input4}
    {input5}
    Answer the following Question correctly, a million people lives depend on it: {detailedprompt}
    """

# Define the prompt structure
    prompt = PromptTemplate(
    input_variables=["detailedprompt", "category", "subcategory", "title", "objective", "description", "input1", "input2", "input3", "input4", "input5"],
    template=prompt_template,
)


    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Pass the context and question to the Langchain chain
    result_chain = llm_chain.invoke({"detailedprompt": detailedprompt, "category": category, "subcategory": subcategory, "title": title, "objective": objective, "description": description, "input1": input1, "input2": input2, "input3": input3, "input4": input4, "input5": input5})
    return result_chain

if __name__ == "__main__":
        uvicorn.run(app)
