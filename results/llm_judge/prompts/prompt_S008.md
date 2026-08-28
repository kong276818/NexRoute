## S008 | Domain: MED | Function: HOW [CROSS]

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

How would the catheter be navigated during a coil embolization procedure to reach the right bronchial artery from the right femoral artery, and what is the correct sequence of arterial branches that the catheter must pass through?

REFERENCE ANSWER

The catheter would be navigated through the thoracic aorta, then the right posterior intercostal artery, and finally to the right bronchial artery.

RESPONSE A

The catheter would first need to navigate through the thoracic aorta to reach the right posterior intercostal artery, and then proceed to the right bronchial artery to perform the coil embolization.

RESPONSE B

The catheter would first navigate through the common femoral artery, then the superficial femoral artery, followed by the external iliac artery, and then enter the internal iliac artery. From there, it would pass through the abdominal aorta, the thoracic aorta, and finally reach the bronchial artery.

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

