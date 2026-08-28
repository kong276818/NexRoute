## S011 | Domain: SOC | Function: WHAT [CROSS]

You are an independent evaluator of two AI-generated answers.

Your task is to evaluate answer quality, not writing-style similarity.

You do not know which system produced either response.
Do not infer or guess system identity.

IMPORTANT EVALUATION PRINCIPLES

1. Judge factual and task-level quality rather than lexical overlap.
2. The reference answer is a guide to the expected substance, not a required wording template.
3. An answer may be correct even if its structure or terminology differs from the reference.
4. Do not reward verbosity.
5. Do not penalize concise answers if they contain the required substance.
6. Do not penalize citation-style or professional-domain formatting by itself.
7. However, if formatting causes substantive information to be missing, unclear, or incorrect,
   penalize the appropriate quality dimension.
8. For cross-domain questions, reward actual integration between domains rather than merely
   listing facts from each domain.
9. Treat Response A and Response B symmetrically.
10. Evaluate each response independently before making a pairwise preference decision.

QUESTION

What cultural and ecological significance does the kiwi hold as New Zealand's national bird?

REFERENCE ANSWER

The kiwi holds significant cultural and ecological importance in New Zealand as the country's national bird. Culturally, it is deeply ingrained in Maori tradition and symbolism, often representing fertility, new life, and the spirit world. Ecologically, it plays a crucial role in seed dispersal and soil health through its feeding habits, making it a vital species for the ecosystem.

RESPONSE A

The kiwi holds significant cultural and ecological significance in New Zealand. Ecologically, it is a unique and endemic bird species that plays a crucial role in the country's biodiversity. Culturally, the kiwi is deeply ingrained in New Zealand's national identity and symbolism, representing resilience, adaptability, and the spirit of the nation. This dual significance underscores the importance of conservation efforts and the role of the kiwi in shaping national pride and environmental stewardship.

RESPONSE B

The kiwi holds significant cultural and ecological importance as New Zealand's national bird. Ecologically, it serves as a symbol of the country's unique biodiversity and contributes to its reputation as a haven for endemic species. Culturally, the kiwi represents national identity, resilience, and the values of New Zealand society, reflecting the nation's pride in its natural heritage and the need to protect its unique ecosystems. Its status as a flightless nocturnal bird also embodies themes of mystery and intrigue, resonating with the public imagination and fostering a sense of national unity around shared environmental stewardship goals.

Evaluate each response on the following 1-5 dimensions:

- factual_correctness
- cross_domain_synthesis  (cross_domain_synthesis: use an integer 1-5 (this IS a cross-domain question))
- task_adequacy
- reasoning_coherence
- completeness
- relevance
- overall_quality

SCORING ANCHORS

5 = excellent / fully satisfies the criterion
4 = good / minor limitations only
3 = acceptable / noticeable limitations
2 = weak / important deficiencies
1 = poor / major failure

PAIRWISE PREFERENCE

Choose: "A", "B", or "TIE"

Choose A or B only when one answer is meaningfully better overall.
Use TIE when differences are small or involve mixed trade-offs.

CONFIDENCE

Rate your confidence in the evaluation: 1 (very uncertain) to 5 (very high confidence).

Return ONLY valid JSON using exactly this schema:

{
  "response_A": {
    "factual_correctness": 1,
    "cross_domain_synthesis": null,
    "task_adequacy": 1,
    "reasoning_coherence": 1,
    "completeness": 1,
    "relevance": 1,
    "overall_quality": 1
  },
  "response_B": {
    "factual_correctness": 1,
    "cross_domain_synthesis": null,
    "task_adequacy": 1,
    "reasoning_coherence": 1,
    "completeness": 1,
    "relevance": 1,
    "overall_quality": 1
  },
  "preference": "TIE",
  "confidence": 1,
  "brief_rationale": "Maximum 60 words."
}

All non-null scores must be integers from 1 to 5.
Do not output markdown.
Do not output commentary outside the JSON.

