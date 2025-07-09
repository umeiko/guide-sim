from openai import OpenAI
from PIL import Image
import numpy as np
import io
import argparse

import base64
# Set OpenAI's API key and API base to use vLLM's API server.
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://192.168.1.110:1234/v1"

class Agent():
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
    
    def chat_with_array(self, image_arr:np.ndarray, prompt:str):
        image_io = io.BytesIO()
        # 将数组转换为 PIL 图像
        image = Image.fromarray(image_arr.astype(np.uint8), 'L')
        # 将图像保存到 BytesIO 对象中
        image.save(image_io, format='PNG')
        image_data = image_io.getvalue()
        # 对图像数据进行 Base64 编码
        base64_encoded_data = base64.b64encode(image_data)
        encoded_image_text = base64_encoded_data.decode("utf-8")
        base64_qwen = f"data:image/png;base64,{encoded_image_text}"

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_qwen
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        return chat_response


if __name__ == "__main__":
    ...