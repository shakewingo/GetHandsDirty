# To allow all LLMs able to call tools. The implementation is only based on the scenario that
# large LLM desn't have ability to call tools directly  but small LLM does, so use large LLM as the planner to decide which tool to use and input params and use small LLM as the executor
# e.g. Qwen2.5-Coder-32B-Instruct cannot call tools while using 

# After testing the script, I think it still has issue that the small LLM can show tool called but cannot return answers properly. One comment in the original video says it's because 
# small LLM cannot call tools either, it's the agent itself able to call tools and the most common scenario is to use small LLM summarize the tool calling thing.

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
import argparse
from typing import Optional, List
from openai import OpenAI
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()
SF_API_KEY = os.getenv('SF_API_KEY')

class ChatCompletionRequest(BaseModel):
    
    model: str
    messages: List
    max_tokens: int = 4096
    temperature: float = 0.7
    tools: Optional[List] = None

# Define args
class Args:
    no_tool_call_base_url = "https://api.siliconflow.com/v1"
    no_tool_call_model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    no_tool_call_api_key = SF_API_KEY
    tool_call_base_url = "https://api.siliconflow.com/v1"
    tool_call_model_name = "Qwen/Qwen2.5-7B-Instruct"
    tool_call_api_key = SF_API_KEY
    host = "172.20.10.2"
    port = 8888

args = Args()
# parser = argparse.ArgumentParser()
# parser.add_argument('--no_tool_call_base_url', type=str, default="https://api.siliconflow.com/v1")
# parser.add_argument('--no_tool_call_model_name', type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
# parser.add_argument('--no_tool_call_api_key', type=str, default=SF_API_KEY)
# parser.add_argument('--tool_call_base_url', type=str, default="https://api.siliconflow.com/v1")
# parser.add_argument('--tool_call_model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
# parser.add_argument('--tool_call_api_key', type=str, default=SF_API_KEY)
# parser.add_argument('--host', type=str, default="172.20.10.2")
# parser.add_argument('--port', type=int, default=8888)

def generate_text(base_url: str, model: str, messages: List, max_tokens: int, temperature: float, api_key: str, tools=None):
    client = OpenAI(base_url=base_url, api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
    )
    return completion


# 定义路由和处理函数，与OpenAI API兼容
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    
    if request.tools:
        TOOL_EAXMPLE = "You will receive a JSON string containing a list of callable tools. Please parse this JSON string and return a JSON object containing the tool name and tool parameters."
 
        REUTRN_FORMAT="{\"tool\": \"tool name\", \"parameters\": {\"parameter name\": \"parameter value\"}}"
        
        INSTRUCTION = f"""
        {TOOL_EAXMPLE}
        Answer the following questions as best you can. 
                
        Use the following format:
        ```tool_json
        {REUTRN_FORMAT}
        ``` 
        
        Please choose the appropriate tool according to the user's question. If you don't need to call it, please reply directly to the user's question. When the user communicates with you in a language other than English, you need to communicate with the user in the same language.
        
        When you have enough information from the tool results, respond directly to the user with a text message without having to call the tool again.
        
        You can use the following tools:
        {request.tools}
        """
        messages = [{"role": "system", "content": INSTRUCTION}]
        messages +=  request.messages
        response = generate_text(args.no_tool_call_base_url, args.no_tool_call_model_name, messages, request.max_tokens, request.temperature, args.no_tool_call_api_key)
        response = response.choices[0].message.content
        print(response)
        print()
    
         # Use the small model to answer based on tool response
        messages = [{"role": "system", "content": "Answer the initial <QUESTION> based on the <INFORMATION> directly."}]
        print(request.messages[-1]['content'] + "\n")
        messages += [{"role": "user", "content": f"<QUESTION>\n{request.messages[-1]['content']}\n</QUESTION>\n<INFORMATION>\n{response}\n</INFORMATION>"}
        ]
        response = generate_text(args.tool_call_base_url, args.tool_call_model_name, messages, request.max_tokens, request.temperature, args.tool_call_api_key, tools=request.tools)
        print(response)
        print()
        
    else:
        response = generate_text(args.no_tool_call_base_url, args.no_tool_call_model_name, request.messages, request.max_tokens, request.temperature, args.no_tool_call_api_key)
    
    return response

if __name__ == "__main__":
    # args = parser.parse_args() # override with common-line args
    uvicorn.run("all2tool_call:app", host=args.host, port=args.port, reload=True)