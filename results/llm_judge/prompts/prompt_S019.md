## S019 | Domain: STEM | Function: WHAT

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

Why would a powered toaster pose the worst consequences if accidentally placed in a full bathtub?

REFERENCE ANSWER

A powered toaster would pose the worst consequences because it contains electrical components that could short-circuit when exposed to water, leading to electric shock, damage to the appliance, and potential fire hazards.

RESPONSE A

A powered toaster poses the worst consequences if accidentally placed in a full bathtub because it increases the risk of electrical shock and drowning simultaneously, making it the most dangerous scenario among the options provided.

RESPONSE B

A powered toaster poses the worst consequences when placed in a full bathtub because it introduces a direct risk of electrocution and potential fire hazards due to the electrical components coming into contact with water. The immersion of an electrical device in water creates conditions for a short circuit, which not only poses a severe electrical shock risk but also can trigger the heating elements to malfunction, increasing the likelihood of a fire starting, thereby endangering anyone nearby who might be in the water.

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

