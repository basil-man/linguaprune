import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from llmlingua import PromptCompressor
from typing import List, Union, Tuple, Dict
from transformers import AutoTokenizer


class AnalyzerSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TokenProbabilityAnalyzer(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
            )
        return cls._instance


class TokenProbabilityAnalyzer(PromptCompressor):
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

    def clean_special_tokens(self, text: str) -> str:
        # 将文本编码成 token id
        input_ids = self.qwen_tokenizer.encode(text, add_special_tokens=False)
        special_ids = set(self.qwen_tokenizer.all_special_ids)

        # 过滤掉特殊 token
        filtered_ids = [tid for tid in input_ids if tid not in special_ids]

        # 解码为文本
        return self.qwen_tokenizer.decode(filtered_ids, skip_special_tokens=True)

    def get_token_importance(
        self, text: str, token_to_word: str = "mean", force_tokens: List[str] = [], force_reserve_digit: bool = False
    ) -> Dict:
        """
        计算文本中每个token的重要性得分，使用与LLMLingua相同的算法。
        """
        # 清理无用 token
        clean_text = self.clean_special_tokens(text)

        # 将文本分成chunks - 正确访问父类的私有方法
        chunks = self._PromptCompressor__chunk_context(clean_text, chunk_end_tokens=set([".", "\n"]))

        # 构造 token_map
        token_map = {}
        for i, t in enumerate(force_tokens):
            if len(self.tokenizer.tokenize(t)) != 1:
                token_map[t] = self.added_tokens[i]

        # 将chunks转换为列表形式以匹配__get_context_prob的输入格式
        context_chunked = [chunks]

        # 计算 token 概率
        probs, _ = self._PromptCompressor__get_context_prob(
            context_chunked,
            token_to_word=token_to_word,
            force_tokens=force_tokens,
            token_map=token_map,
            force_reserve_digit=force_reserve_digit,
        )

        return probs


# 配置日志
logging.basicConfig(level=logging.INFO)

# 创建应用
app = FastAPI()

analyzer = AnalyzerSingleton.get_instance()
device = next(analyzer.model.parameters()).device
logging.info(f"Analyzer model loaded on: {device}")


# 请求模型
class SolutionRequest(BaseModel):
    solution_str: str


@app.post("/get_token_importance")
async def get_token_importance(req: SolutionRequest):
    try:
        importance = analyzer.get_token_importance(req.solution_str)[0]
        return {"importance": float(importance) if isinstance(importance, (int, float)) else importance}
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
