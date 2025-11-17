"""
Data models: QuestionItem and loader for the 20 evaluation questions.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class QuestionItem:
    id: int
    question: str
    gold_answer: str


def load_default_questions() -> List[QuestionItem]:
    data = [
        {
            "id": 1,
            "question": "What is the main research question of the thesis?",
            "gold_answer": "The thesis investigates how differential privacy–driven text rewriting can be systematically integrated into multiple stages of a RAG pipeline to reduce sensitive information leakage while maintaining retrieval accuracy and generation quality."
        },
        {
            "id": 2,
            "question": "What is the title of the thesis?",
            "gold_answer": "Deep Comparative Evaluation of Differential Privacy Integration Across Multi-Phase Retrieval-Augmented Generation Pipelines."
        },
        {
            "id": 3,
            "question": "What datasets are used to evaluate the proposed framework?",
            "gold_answer": "The thesis uses two datasets: the Enron Email dataset and the HealthcareMagic-101 dataset."
        },
        {
            "id": 4,
            "question": "What are the motivations for studying differential privacy in RAG systems?",
            "gold_answer": "RAG systems rely on sensitive domain-specific data, and existing mitigation techniques like summarization or retrieval filtering do not fully prevent leakage. Differential privacy offers quantifiable guarantees to limit leakage risk."
        },
        {
            "id": 5,
            "question": "Which RAG limitations motivate DP integration?",
            "gold_answer": "RAG mitigates hallucinations and outdated knowledge but introduces privacy vulnerabilities, including prompt-based extraction attacks and risks from cloud-hosted LLM logging."
        },
        {
            "id": 6,
            "question": "What privacy-preserving methods does prior research commonly apply in RAG?",
            "gold_answer": "Prior work uses pre-retrieval filtering, post-retrieval summarization or paraphrasing, and instruction-based defenses, but they only partially reduce leakage risks."
        },
        {
            "id": 7,
            "question": "What are the main differential privacy mechanisms relevant to this work?",
            "gold_answer": "Differential privacy mechanisms include Laplace noise, Gaussian noise, DP-SGD, and text rewriting models such as DP-BART and DP-MLM."
        },
        {
            "id": 8,
            "question": "What gap in the literature does this thesis aim to address?",
            "gold_answer": "The thesis provides the first comprehensive stage-wise evaluation of DP-based text rewriting applied across different stages of the RAG pipeline, addressing gaps in prior research that only proposed integration points without systematic comparison."
        },
        {
            "id": 9,
            "question": "What stages of the RAG pipeline are considered for DP integration?",
            "gold_answer": "DP is evaluated at three stages: database-level rewriting, pre-generation rewriting, and output-stage rewriting, under both local and cloud LLM configurations."
        },
        {
            "id": 10,
            "question": "What is the core problem statement of the thesis?",
            "gold_answer": "The core problem is how to preserve retrieval accuracy and generation quality while applying differential privacy–based text rewriting across multiple RAG stages to reduce sensitive information leakage."
        },
        {
            "id": 11,
            "question": "What research objectives are outlined in the proposal?",
            "gold_answer": "Objectives include analyzing DP mechanisms, applying DP rewriting at multiple RAG stages, evaluating privacy–utility tradeoffs, identifying optimal integration strategies, and giving guidelines for local and cloud RAG deployments."
        },
        {
            "id": 12,
            "question": "In which domains is privacy especially important for RAG systems?",
            "gold_answer": "Privacy is especially important in healthcare, legal advisory, and financial domains where sensitive information is commonly used."
        },
        {
            "id": 13,
            "question": "What types of attacks threaten the privacy of RAG systems?",
            "gold_answer": "Attacks include prompt-based extraction attacks, membership inference attacks, and both retrieval and generation leakage of sensitive records."
        },
        {
            "id": 14,
            "question": "How does the thesis quantify privacy leakage?",
            "gold_answer": "The thesis quantifies privacy through differential privacy’s privacy budget epsilon (ε), evaluating leakage risk under different DP mechanisms."
        },
        {
            "id": 15,
            "question": "How does the thesis measure utility preservation?",
            "gold_answer": "Utility is measured through retrieval accuracy, generation quality, and preservation of semantic meaning after DP text rewriting."
        },
        {
            "id": 16,
            "question": "What is DP-BART?",
            "gold_answer": "DP-BART is a text rewriting model trained under local differential privacy that injects calibrated noise during rewriting to preserve meaning while guaranteeing privacy."
        },
        {
            "id": 17,
            "question": "Why is unstructured text challenging for differential privacy?",
            "gold_answer": "Unstructured text lacks formal structure, making noise addition difficult without harming meaning; DP must balance semantic preservation and privacy guarantees."
        },
        {
            "id": 18,
            "question": "Why is LLaMA mentioned in the privacy-related discussion?",
            "gold_answer": "LLaMA is cited as a model where instruction-based defenses performed strongly, highlighting that model-specific protections exist but are not sufficient for principled privacy guarantees."
        },
        {
            "id": 19,
            "question": "What outputs does the thesis aim to deliver?",
            "gold_answer": "The thesis aims to deliver a DP-integrated RAG framework, quantitative evaluation results, and guidelines for selecting DP mechanisms and deployment strategies."
        },
        {
            "id": 20,
            "question": "Who are the student and advisors listed in the proposal?",
            "gold_answer": "The student is Zainab Saad (ID 202472448). Advisors include Prof. Ibrahim Issa, Prof. Khalil Hariss, and committee member Prof. Razane Tajeddine."
        },
    ]
    return [QuestionItem(**d) for d in data]
