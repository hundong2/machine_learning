from googlelangchainExample import get_model as get_google_model
from langchainExample import get_model as get_openai_model
import random
import json
if random.random() < 0.5: # 50% probability
    model = get_google_model()
    print("google model")
else:
    model = get_openai_model()
    print("local openai model")

result = model.invoke("Hello, world!")
print(json.dumps(result.content, indent=4))