## S020 | Domain: STEM | Function: WHAT

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

How does the formation of the penumbra occur during an eclipse, and what role does it play in the overall shadow structure?

REFERENCE ANSWER

During an eclipse, the penumbra forms as the outer part of the shadow where only a portion of the light source is blocked by the eclipsing body. This occurs because the light from the unobstructed parts of the light source still reaches these areas, creating a partial illumination. The penumbra plays a crucial role in the overall shadow structure by delineating the boundary between the fully illuminated and completely shadowed regions, thus providing a visual transition zone.

RESPONSE A

During an eclipse, particularly a solar eclipse, the penumbra forms as the primary shadow cast by the Moon onto the Earth, marking the region where the Sun is only partially blocked, leading to a partial solar eclipse for observers within this area. This penumbral shadow plays a crucial role in shaping the overall shadow structure during an eclipse, delineating the boundary between areas experiencing a total eclipse where the Sun is completely obscured and those seeing only a partial eclipse, thus defining the transition zone between full darkness and partial illumination.

RESPONSE B

The penumbra forms when some but not all of the sun's light is blocked by the moon, creating a partial shadow. It plays a significant role in the overall shadow structure by defining the boundary between full darkness (umbra) and partial darkness (penumbra).

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

