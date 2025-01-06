import os
from anthropic import Anthropic

client = Anthropic()
prompt = "Where's the capital of France?"
response = client.messages.create(model="claude-3-haiku-20240307",
                                  max_tokens=1000,
                                  temperature=0,
                                  messages=[{
                                      "role": "user",
                                      "content": prompt
                                  }]
                                  )

print(response.content[0].text)
