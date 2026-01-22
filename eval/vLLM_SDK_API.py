import asyncio
import httpx
from typing import List
from openai import AsyncOpenAI, OpenAIError
from transformers import AutoTokenizer

class VLLM_API_V1:
    """
    vLLM Completions API(/v1/completions)와 상호작용하기 위한 확장 가능한 클라이언트 (v1 로직 기반).

    - 초기화 시 토크나이저와 HTTP 클라이언트를 한 번만 생성하여 재사용합니다.
    - 클래스 기반으로 설계되어 외부 모듈에서 import하여 사용하기 용이합니다.
    """
    def __init__(self, model_path: str, base_url: str = "http://localhost:8005/v1", served_model_name: str = None):
        """
        VLLM_API_V1 클라이언트를 초기화합니다.

        :param model_path: Hugging Face 모델 경로 또는 로컬 경로. 토크나이저 로딩에 사용됩니다.
        :param base_url: vLLM 서버의 OpenAI 호환 API 엔드포인트 주소.
        :param served_model_name: vLLM 구동 시 --served-model-name으로 지정한 모델 이름.
                                  None일 경우 model_path의 마지막 부분을 사용합니다.
        """
        print(f"Initializing VLLM_API_V1 client for model: {model_path}")
        self.served_model_name = served_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        limits = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
        self.client_async = AsyncOpenAI(
            base_url=base_url,
            api_key="not-used",
            #! 이때 timeout 오류가 나지 않도록 충분한 timeout 시간 할당해 주자.
            http_client=httpx.AsyncClient(http2=True, limits=limits, timeout=30000)
        )
        print(f"Client initialized. Target model on server: {self.served_model_name}")

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        stop_tokens: List[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        top_p: float = 0.6,
        stream: bool = False,
        tokenizer_args: dict = None,
        completion_api_args: dict = None,
        **kwargs
    ) -> str:
        """
        주어진 프롬프트를 기반으로 텍스트를 생성합니다. (v1 로직)
        """
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]
        
        # 1. 기본 extra_body 값 설정
        final_extra_body = {"skip_special_tokens": False}

        # 2. 사용자가 추가 인자를 전달한 경우, 기본값에 덮어쓰며 병합
        # TODO : 덮어쓰는 방식이 아니라 추가하는 방식으로 하고, 만약 사용자가 skip_special_tokens를 이미 구성했을 경우 그때 사용자 것을 사용하도록 구성.
        if completion_api_args:
            final_extra_body.update(completion_api_args)

        try:
            prompt_str = self.tokenizer.apply_chat_template(
                    chat_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **(tokenizer_args or {}),
                )
        except Exception as e:
            return f"Error applying chat template: {e}"

        try:
            response = await self.client_async.completions.create(
                model=self.served_model_name,
                prompt=prompt_str,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                stop=stop_tokens,
                extra_body=final_extra_body,
            )
            
            if stream:
                # 스트리밍 응답 처리는 이 예제에서 생략합니다.
                return "Streaming response is not handled in this basic example."
            else:
                return response.choices[0].text.strip()

        except OpenAIError as e:
            print(f"An API error occurred: {e}")
            return f"Error: {e}"

if __name__ == '__main__':
    
    # --- 설정 변수 ---
    # ❗️ 사용 전 이 부분을 자신의 환경에 맞게 수정하세요.
    MODEL_PATH = "LGAI-EXAONE/EXAONE-4.0-32B"  # 로컬 모델 경로 또는 Hugging Face 모델 식별자
    BASE_URL = "http://localhost:8005/v1"
    SERVED_MODEL_NAME = "exaone-4-32b" # vLLM 구동 시 지정한 --served-model-name
    
    async def run_test():
        print("--- Creating VLLM API Client instance ---")
        # 1. 클래스 인스턴스 생성
        api_client = VLLM_API_V1(
            model_path=MODEL_PATH,
            base_url=BASE_URL,
            served_model_name=SERVED_MODEL_NAME
        )
        
        print("\n--- Sending a request ---")
        # 2. 인스턴스의 메서드 호출
        test_prompt = "대한민국의 인공지능 기술 현황에 대해 알려줘."
        response_text = await api_client.generate(prompt=test_prompt)
        
        print("\n--- Generated Response ---")
        print(response_text)
        print("--------------------------")

    asyncio.run(run_test())