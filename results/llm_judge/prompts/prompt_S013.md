## S013 | Domain: SOC | Function: WHAT

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

What is the diaphragm and what role does it play in the respiratory system?

REFERENCE ANSWER

The diaphragm is a large, dome-shaped muscle located at the base of the lungs. It plays a crucial role in the respiratory system by contracting and relaxing to facilitate breathing. During inhalation, the diaphragm contracts and moves downward, increasing the volume of the chest cavity and allowing air to flow into the lungs under pressure from the expanding abdominal muscles. During exhalation, the diaphragm relaxes and moves upward, decreasing the volume of the chest cavity and expelling air from the lungs.

RESPONSE A

The diaphragm is a dome-shaped skeletal muscle located beneath the thoracic cavity and above the abdominal cavity, acting as the primary muscle of respiration. Its contraction and relaxation facilitate the movement of air in and out of the lungs by altering the volume of the thoracic cavity. Specifically, during inhalation, the diaphragm contracts and descends, increasing the volume of the thoracic cavity and lowering intra-thoracic pressure, which draws air into the lungs; conversely, during exhalation, the diaphragm relaxes and ascends, decreasing the volume of the thoracic cavity and increasing intra-thoracic pressure, aiding in the expulsion of air from the lungs. This vital function of the diaphragm ensures efficient gas exchange and maintains the necessary pressure gradients for breathing, underscoring its crucial role in supporting both the respiratory and digestive systems through their anatomical separation.

RESPONSE B

The diaphragm is a dome-shaped muscle located at the base of the lungs. It plays a crucial role in respiration by contracting and relaxing to facilitate the movement of air in and out of the lungs. When the diaphragm contracts, it flattens and moves downward, increasing the volume of the thoracic cavity and allowing air to flow into the lungs. Conversely, when the diaphragm relaxes, it returns to its dome shape, decreasing the volume of the thoracic cavity and forcing air out of the lungs. This process is essential for breathing and maintaining proper gas exchange in the body.

Evaluate each response on the following 1-5 dimensions:

- factual_correctness
- cross_domain_synthesis  (cross_domain_synthesis: set to null (this is a single-domain question))
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

