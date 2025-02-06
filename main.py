import orjson
import fire
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


class Item(BaseModel):
    origin_context: str = Field(description="번역 전 [맥락] 입력 값")
    context_translated: str = Field(description="[맥락] 입력을 [번역 결과 언어]로 번역한 결과 값")
    origin: str = Field(description="번역 전 [번역할 내용] 입력 값")
    translated: str = Field(description="[번역할 내용] 입력을 [번역 결과 언어]로 번역한 결과 값")


user_prompt_template = """아래의 입력 데이터를 {target_lang}로 번역하여, 반드시 아래 JSON 형식과 동일하게 출력해 주세요.

[입력 데이터]
{{
  "origin_context": "{context}",
  "origin": "{content}"
}}

[요청 사항]
- "origin_context" 항목은 번역 전 맥락입니다.
- "origin" 항목은 번역 전 내용입니다.
- 각각을 {target_lang}로 번역하여,
    - 번역된 맥락은 "context_translated"에,
    - 번역된 내용은 "translated"에 입력해 주세요.

**반드시 아래 JSON 구조로 출력해 주세요:**
{{
  "origin_context": "<번역 전 맥락 원본>",
  "context_translated": "<번역된 맥락>",
  "origin": "<번역 전 내용 원본>",
  "translated": "<번역된 내용>"
}}
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
        key: str = "answer",
    ):
        with open(filename, mode="r", encoding="utf-8") as f:
            rows = orjson.loads(f.read())

        result = []
        if batch:
            for i in tqdm(range(0, len(rows), batch_size)):
                batch = rows[i : i + batch_size]
                inputs = [{"context": row["prompt"], "content": row[key], "target_lang": target_lang} for row in batch]
                responses = self.chain.batch(inputs)
                result.extend([response.model_dump() for response in responses])
        else:
            for i in tqdm(range(0, len(rows), batch_size)):
                context = rows[i]["prompt"]
                content = rows[i][key]
                response = self.chain.invoke({"context": context, "content": content, "target_lang": target_lang})
                result.append(response.model_dump())
            with open(output, "w", encoding="utf-8") as f:
                f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode("utf-8"))

        with open(output, "w", encoding="utf-8") as f:
            f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode("utf-8"))


if __name__ == "__main__":
    fire.Fire(Cli)
