import os
from dotenv import load_dotenv
load_dotenv()
import requests
from ragas.metrics.collections import Faithfulness, ContextRecall, AnswerCorrectness
from src.utils.llm_config_utils.llm_initiate import (
    authenticate_llm_gateway
)
from src.utils.llm_config_utils.llm_setup_for_ragas import (

    build_async_openai_client,
    build_sync_openai_client,
    build_ragas_llm,
)

class RagasConfig:
    def __init__(self, question, answer, contexts, ref_str):
        self.question = question
        self.answer = answer
        self.contexts = contexts
        self.ref_str = ref_str
# Setup clients
    def get_llm(self):
        model = os.environ.get("LLM_MODEL", "gpt-4")
        openai_url = os.environ.get("LLM_OPENAI_URL", "")
        base_url = f"{openai_url}/openai/deployments/{model}"
        token = authenticate_llm_gateway()
        api_version = os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview")
        async_client = build_async_openai_client(token, base_url, api_version)
        llm = build_ragas_llm(async_client, model)
        return llm

    def get_evaluation_metrics(self):
        llm = self.get_llm()
        faithfulness_result = self.get_faithfulness_score(llm)
        print(f"Faithfulness Score: {faithfulness_result}")

        # ContextRecall.score expects user_input, retrieved_contexts, and reference (str)
        ctxrecall_result = self.get_context_recall_score(llm=llm)
        print(f"Context Recall Score: {ctxrecall_result}")

        # AnswerCorrectness.score expects user_input, response, and reference (str)
        ans_correctness_result = self.get_answer_correctness_score(llm=llm)
        print(f"Answer Correctness Score: {ans_correctness_result}")

        return ({'metrics':(f"faithfulness: {faithfulness_result}",
                    f"context_recall: {ctxrecall_result}",
                    f"answer_correctness: {ans_correctness_result}"
                    )})

    def get_faithfulness_score(self, llm):
        faith_scorer = Faithfulness(llm=llm)
        # Faithfulness.score expects user_input, response, and retrieved_contexts
        faith_result = faith_scorer.score(
            user_input=self.question,
            response=self.answer,
            retrieved_contexts=self.contexts,
        )
        return faith_result.value

    def get_context_recall_score(self, llm):
        ctxrecall_scorer = ContextRecall(llm=llm)
        ctxrecall_result = ctxrecall_scorer.score(
            user_input=self.question,
            retrieved_contexts=self.contexts,
            reference=self.ref_str,
        )
        return ctxrecall_result.value
    
    def get_answer_correctness_score(self, llm):
        ans_correctness_scorer = AnswerCorrectness(llm=llm, weights=[1.0, 0.0])
        ans_corr_result = ans_correctness_scorer.score(
            user_input=self.question,
            response=self.answer,
            reference=self.ref_str,
        )
        return ans_corr_result.value
