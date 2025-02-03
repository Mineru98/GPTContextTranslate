import orjson
import fire
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


class Item(BaseModel):
    origin: str = Field(description="번역 전 입력 값")
    translated: str = Field(description="번역 후 결과 값")


user_prompt_template = """### 맥락

{context}

### 번역할 내용

{content}

### 요청 사항:
 
- 주어진 [맥락] 다음에 오는 문장을 {target_lang}로 번역해 주세요.
- [맥락] 부분도 번역해 주세요.
- 번역 시 JSON 형식으로 반환해 주세요.
"""
model_name = "gpt-4o-mini-2024-07-18"
combined_prompt = PromptTemplate.from_template(user_prompt_template)


class Cli:
    def __init__(self):
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.32,
            max_tokens=128,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        ).with_structured_output(Item)
        self.chain = combined_prompt | llm

    def run(
        self,
        filename: str = "score.json",
        output: str = "translate.json",
        target_lang: str = "한국어",
        batch: bool = True,
        batch_size: int = 10,
    ):
        with open(filename, mode="r", encoding="utf-8") as f:
            rows = orjson.loads(f.read())

        result = []
        if batch:
            for i in tqdm(range(0, len(rows), batch_size)):
                batch = rows[i : i + batch_size]
                inputs = [{"context": row["prompt"], "content": row["answer"], "target_lang": target_lang} for row in batch]
                responses = self.chain.batch(inputs)
                result.extend([response.model_dump() for response in responses])
        else:
            for i in tqdm(range(0, len(rows), batch_size)):
                context = rows[i]["prompt"]
                content = rows[i]["answer"]
                response = self.chain.invoke({"context": context, "content": content, "target_lang": target_lang})
                result.append(response.model_dump())

        with open(output, "w", encoding="utf-8") as f:
            f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode("utf-8"))


if __name__ == "__main__":
    fire.Fire(Cli)
