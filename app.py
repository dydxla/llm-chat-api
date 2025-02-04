from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# 사용할 모델 이름 설정 (원하는 모델로 변경 가능)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# 토크나이저와 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # GPU 메모리 효율화를 위해
    device_map="auto"
)
model.eval()  # 평가 모드

# 각 메시지 구조 정의
class Message(BaseModel):
    role: str   # "user" 또는 "assistant" 등 (필요에 따라 "system" 등 추가 가능)
    content: str

# 대화 요청 데이터 모델
class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")
def chat(request: ChatRequest):
    # 대화 내역을 하나의 프롬프트로 변환합니다.
    # 간단한 예시로 "User:"와 "Assistant:" 태그를 붙여서 구성합니다.
    prompt = ""
    for message in request.messages:
        if message.role.lower() == "user":
            prompt += f"User: {message.content}\n"
        elif message.role.lower() == "assistant":
            prompt += f"Assistant: {message.content}\n"
        else:
            prompt += f"{message.role}: {message.content}\n"
    # 모델에게 다음에 이어질 답변을 생성하도록 유도
    prompt += "Assistant:"

    # 프롬프트 토크나이징 및 생성 (torch.no_grad() 사용)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 1024,  # 기존 프롬프트 + 최대 100 토큰 생성
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id  # 필요한 경우 추가
        )
    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 프롬프트와 생성된 결과의 차이를 추출해 응답 메시지로 활용할 수 있음
    # 여기서는 간단하게 전체 텍스트를 반환합니다.
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
