# LLM-Judge 평가 프롬프트

각 섹션의 프롬프트를 Gemini / Claude / GPT-4 웹 UI에 붙여넣고 JSON 응답을 score_sheet.csv에 기록하세요.

---

## [01_LAW_NexRoute_v01] Domain: LAW

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** How do state court decisions typically reconcile the application of state statutes prohibiting employment discrimination with the National Labor Relations Act (NLRA) when such statutes are not preempted by the NLRA?

**Reference Answer:** State court decisions generally hold that the National Labor Relations Act (NLRA) does not preempt proceedings under state statutes prohibiting employment discrimination, such as those based on race, sex, religion, or other similar grounds. This means that employees can pursue claims under both state anti-discrimination laws and the NLRA, and state courts often recognize the concurrent jurisdiction and complementary nature of these laws, allowing for the assertion of discrimination claims even if there is a collective bargaining agreement that addresses related issues.

**System Answer:** The correct holding is option 1: holding that state law claims were not preempted because they did not require interpretation of the collective bargaining agreement

All holdings:
0: holding that the plaintiffs claims were preempted by the labor management relations act
1: holding that state law claims were not preempted because they did not require interpretation of the collective bargaining agreement
2: holding that state law claims are preempted where the resolution of the dispute necessarily requires an interpretation of the collective bargaining agreement
3: holding that a plaintiffs state law claims were not preempted by the labor management relations act because the plaintiffs complaint did not seek to compel arbitration under the terms of the parties collective bargaining agreement
4: holding that the plaintiffs state law claims were not preempted because they did not require interpretation of the collective bargaining agreement

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [01_LAW_NaiveRAG_b3] Domain: LAW

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** How do state court decisions typically reconcile the application of state statutes prohibiting employment discrimination with the National Labor Relations Act (NLRA) when such statutes are not preempted by the NLRA?

**Reference Answer:** State court decisions generally hold that the National Labor Relations Act (NLRA) does not preempt proceedings under state statutes prohibiting employment discrimination, such as those based on race, sex, religion, or other similar grounds. This means that employees can pursue claims under both state anti-discrimination laws and the NLRA, and state courts often recognize the concurrent jurisdiction and complementary nature of these laws, allowing for the assertion of discrimination claims even if there is a collective bargaining agreement that addresses related issues.

**System Answer:** State court decisions often balance the application of state statutes prohibiting employment discrimination with the National Labor Relations Act (NLRA) by interpreting the statutes in a manner that aligns with the goals of the NLRA, particularly those related to collective bargaining and workers' rights. When state statutes are not preempted by the NLRA, courts may consider the purpose behind both sets of laws and strive to achieve a harmonious interpretation that protects employees from discrimination while also respecting their right to engage in concerted activities for mutual aid or protection.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [02_LAW_NexRoute_v01] Domain: LAW

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the legal principles underlying the holding in the Gold Answer with those in holding number 1, focusing on the criteria for determining whether a case remains justiciable despite the cessation of the challenged action.

**Reference Answer:** The Gold Answer holds that a case was not moot where the city failed to provide assurance that the policeman charged with misconduct would not be rehired or that the challenged conduct would not be resumed, indicating that the justiciability of the case depends on the likelihood of future recurrence of the challenged action. Holding number 1, on the other hand, states that voluntary cessation of a challenged action does not make a case moot unless the defendant can show that there is no reasonable expectation that the challenged action will be repeated, suggesting that the case must remain justiciable if there is a reasonable expectation of recurrence.

**System Answer:** The correct holding is option 0: holding that a dismissal without prejudice for failure to pay filing fees is not a final judgment

All holdings:
0: holding that a dismissal without prejudice for failure to pay filing fees is not a final judgment
1: holding that a dismissal without prejudice for failure to state a claim is not a final judgment
2: holding that a dismissal for failure to state a claim is a final judgment
3: holding that a dismissal with prejudice is a final judgment
4: holding that a dismissal with prejudice is a final judgment

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [02_LAW_NaiveRAG_b3] Domain: LAW

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the legal principles underlying the holding in the Gold Answer with those in holding number 1, focusing on the criteria for determining whether a case remains justiciable despite the cessation of the challenged action.

**Reference Answer:** The Gold Answer holds that a case was not moot where the city failed to provide assurance that the policeman charged with misconduct would not be rehired or that the challenged conduct would not be resumed, indicating that the justiciability of the case depends on the likelihood of future recurrence of the challenged action. Holding number 1, on the other hand, states that voluntary cessation of a challenged action does not make a case moot unless the defendant can show that there is no reasonable expectation that the challenged action will be repeated, suggesting that the case must remain justiciable if there is a reasonable expectation of recurrence.

**System Answer:** The legal principles underlying the holding in the Gold Answer and holding number 1 both emphasize the importance of the continuing effects of the challenged action on the plaintiff's rights. However, they differ in their specific criteria for determining whether a case remains justiciable. The Gold Answer focuses on the need for a live controversy and the ability to provide effective relief, while holding number 1 emphasizes the irreparable harm and the need for a concrete and particularized injury. These differences highlight the nuanced approach required in legal analysis and the varying standards applied in different jurisdictions.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [03_LAW_NexRoute_v01] Domain: LAW

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** What legal principle does the War Crimes Act of 1996 seek to enforce through its definition of ‘war crime’, and how does it relate to the international law of war?

**Reference Answer:** The War Crimes Act of 1996 seeks to enforce the legal principle that certain acts committed during warfare, as prescribed by the international law of war, constitute war crimes. This act defines war crimes to include violations of the law of nations or a protocol or other agreement concerning the law of war to which the United States is a party, thereby aligning domestic law with international legal standards governing the conduct of war and the treatment of individuals and nations during conflict.

**System Answer:** The correct holding is option 1: holding that the war crimes act of 1996 is an exercise of congresss enumerated power to make rules concerning captures on land and water

All holdings:
0: holding that the war crimes act of 1996 is an exercise of congresss enumerated power to define and punish offenses against the law of nations
1: holding that the war crimes act of 1996 is an exercise of congresss enumerated power to make rules concerning captures on land and water
2: holding that congresss exercise of the power to regulate interstate commerce includes the power to regulate the channels through which commerce flows
3: holding that congress may exercise its power to tax in order to fund its exercise of other enumerated powers
4: holding that congress may exercise its power to regulate interstate commerce by regulating the channels through which commerce flows

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [03_LAW_NaiveRAG_b3] Domain: LAW

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** What legal principle does the War Crimes Act of 1996 seek to enforce through its definition of ‘war crime’, and how does it relate to the international law of war?

**Reference Answer:** The War Crimes Act of 1996 seeks to enforce the legal principle that certain acts committed during warfare, as prescribed by the international law of war, constitute war crimes. This act defines war crimes to include violations of the law of nations or a protocol or other agreement concerning the law of war to which the United States is a party, thereby aligning domestic law with international legal standards governing the conduct of war and the treatment of individuals and nations during conflict.

**System Answer:** The War Crimes Act of 1996 enforces the legal principle of prohibiting certain acts during wartime, specifically those that violate the international law of war. It defines ‘war crime’ to include grave breaches of the Geneva Conventions and other serious violations of the laws and customs of war. This relates to the international law of war by reinforcing the legal framework established under the Geneva Conventions and other international treaties, ensuring accountability for war crimes and promoting adherence to humanitarian standards in armed conflicts.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [04_LAW_NexRoute_v01] Domain: LAW

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** How does a court determine if an employer's proffered reason for an employment action is pretextual according to the holding in Grayson v. O'Neill?

**Reference Answer:** According to the holding in Grayson v. O'Neill, a plaintiff can establish pretext only by showing that the employer's explanation is a hollow one, lacking in substance and not genuinely believed.

**System Answer:** The correct holding is option 1: holding that the proper inquiry is whether the plaintiffs proffered reason is so unpersuasive and implausible that no reasonable person could believe it

All holdings:
0: holding that a district court must examine the totality of the circumstances to determine whether a reasonable person could have believed that the defendant was authorized to enter the premises
1: holding that the proper inquiry is whether the plaintiffs proffered reason is so unpersuasive and implausible that no reasonable person could believe it
2: holding that a reasonable jury could find that the plaintiffs proffered reason for its adverse action against the plaintiff was pretextual where the plaintiff presented evidence that the employer was motivated by a desire to discriminate against him because of his race
3: holding that an employee may rebut the employers proffered nondiscriminatory reason for an adverse employment action by demonstrating that either 1 the employer harbored discriminatory animus or 2 the employers proffered reason is pretextual internal quotation marks omitted
4: holding that a plaintiff may establish a violation of the adea by demonstrating that the employers asserted justification for its action is pretextual

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [04_LAW_NaiveRAG_b3] Domain: LAW

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** How does a court determine if an employer's proffered reason for an employment action is pretextual according to the holding in Grayson v. O'Neill?

**Reference Answer:** According to the holding in Grayson v. O'Neill, a plaintiff can establish pretext only by showing that the employer's explanation is a hollow one, lacking in substance and not genuinely believed.

**System Answer:** According to the holding in Grayson v. O'Neill, a court determines if an employer's proffered reason for an employment action is pretextual by examining whether the employer's explanation lacks credibility or if there is evidence suggesting a discriminatory motive behind the action. The court looks at whether the employer's stated reason was actually the true reason for the decision or if it was merely a pretext to hide a discriminatory purpose.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [05_LAW_NexRoute_v01] Domain: LAW

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the legal significance of the Douglas analysis in Pleener's case with the holding in option 0, specifically addressing how they differ in their treatment of the burden of proof and the demonstration of pretext in employment discrimination cases.

**Reference Answer:** The Douglas analysis in Pleener's case concluded that even if the plaintiff could demonstrate that the defendant's proffered nondiscriminatory reason for removal (in this case, a failed drug test) was in error, such a showing would not demonstrate that the reliance on the test was pretext for discrimination. This holding emphasizes the need for the plaintiff to prove both that the proffered reason is false and that the employer’s true reason was discriminatory. In contrast, option 0 also holds that even if the plaintiff can show that the proffered reason was in error, this alone does not establish that the reliance on the test was a pretext for discrimination. Both analyses highlight the stringent requirement for the plaintiff to demonstrate both the falsity of the proffered reason and that the true motive was discriminatory, rather than merely establishing an error in the proffered reason.

**System Answer:** The correct holding is option 1: holding that the burden of proving that the employers proffered reasons for its actions were pretextful is on the plaintiff

All holdings:
0: holding that the burden of proving that the employers proffered reasons for its actions were pretextful is on the plaintiff
1: holding that the burden of proving that the employers proffered reasons for its actions were pretextful is on the plaintiff
2: holding that the burden of proving pretext is on the plaintiff
3: holding that the burden is on the plaintiff
4: holding that the burden is on the plaintiff

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [05_LAW_NaiveRAG_b3] Domain: LAW

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the legal significance of the Douglas analysis in Pleener's case with the holding in option 0, specifically addressing how they differ in their treatment of the burden of proof and the demonstration of pretext in employment discrimination cases.

**Reference Answer:** The Douglas analysis in Pleener's case concluded that even if the plaintiff could demonstrate that the defendant's proffered nondiscriminatory reason for removal (in this case, a failed drug test) was in error, such a showing would not demonstrate that the reliance on the test was pretext for discrimination. This holding emphasizes the need for the plaintiff to prove both that the proffered reason is false and that the employer’s true reason was discriminatory. In contrast, option 0 also holds that even if the plaintiff can show that the proffered reason was in error, this alone does not establish that the reliance on the test was a pretext for discrimination. Both analyses highlight the stringent requirement for the plaintiff to demonstrate both the falsity of the proffered reason and that the true motive was discriminatory, rather than merely establishing an error in the proffered reason.

**System Answer:** The Douglas analysis in Pleener's case emphasizes the need for the plaintiff to demonstrate a prima facie case of discrimination and then shift the burden of proof to the defendant to show a legitimate reason for the adverse action. This differs from option [1], which may involve a different approach to shifting the burden of proof or a more stringent requirement for the plaintiff to establish the existence of a discriminatory motive. Both analyses aim to protect employees from discriminatory practices but may vary in their procedural requirements and the weight given to each party's burden.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [06_MED_NexRoute_v01] Domain: MED

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the evidence for progressive structural brain abnormalities in schizophrenia with the findings regarding progressive medial temporal lobe involvement, considering the role of patient age and illness duration in these outcomes.

**Reference Answer:** These findings suggest that schizophrenia is associated with progressive structural brain abnormalities, affecting both gray and white matter. However, we found no evidence to suggest progressive medial temporal lobe involvement but did find evidence that this may be partly explained by heterogeneity between studies in patient age and illness duration. The causes and clinical correlates of these progressive brain changes should now be the focus of investigation.

**System Answer:** The evidence suggests that schizophrenia is associated with progressive structural brain abnormalities, impacting both gray and white matter regions, yet it explicitly notes the absence of progressive medial temporal lobe involvement. This distinction underscores the complexity of brain changes in schizophrenia, highlighting that while while progressive alterations occur across various brain areas, the medial temporal lobes, including critical structures such as the hippocampus and amygdala, remain relatively stable. The lack of medial temporal lobe involvement could be attributed to the heterogeneity in patient age and illness duration among studied populations, suggesting that varying stages of the disease and individual developmental trajectories influence the extent and pattern of brain changes, thereby emphasizing the need for further investigation into understanding the underlying mechanisms and clinical correlates of these progressive brain alterations.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [06_MED_NaiveRAG_b3] Domain: MED

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the evidence for progressive structural brain abnormalities in schizophrenia with the findings regarding progressive medial temporal lobe involvement, considering the role of patient age and illness duration in these outcomes.

**Reference Answer:** These findings suggest that schizophrenia is associated with progressive structural brain abnormalities, affecting both gray and white matter. However, we found no evidence to suggest progressive medial temporal lobe involvement but did find evidence that this may be partly explained by heterogeneity between studies in patient age and illness duration. The causes and clinical correlates of these progressive brain changes should now be the focus of investigation.

**System Answer:** Both the evidence for progressive structural brain abnormalities in schizophrenia and the findings regarding progressive medial temporal lobe involvement highlight the importance of considering patient age and illness duration in understanding these outcomes. However, the nature of the abnormalities differs; schizophrenia is characterized by widespread gray matter loss across multiple regions, while medial temporal lobe involvement suggests a more localized pattern of atrophy. These differences have significant implications for both the pathophysiology of these conditions and the clinical management strategies employed.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [07_MED_NexRoute_v01] Domain: MED

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Why might the lack of difference between the thyroxine treatment and placebo groups in the study suggest a latency period rather than other potential biases or explanations?

**Reference Answer:** The lack of difference between the thyroxine treatment and placebo groups in the study suggests a latency period because the therapeutic effects of thyroxine in asymptomatic subclinical hypothyroidism may take longer than the 3-week follow-up period to manifest, thus not showing any significant improvement during this short observation time.

**System Answer:** The lack of difference between the thyroxine treatment and placebo groups in the study suggests a latency period rather than potential biases or explanations because it indicates that the physiological changes induced by thyroxine require time to manifest fully, highlighting the delayed onset of effects associated with latency periods. This phenomenon underscores the complexity of biological responses to hormone treatments, where immediate changes may not be observable, necessitating longer observation periods to accurately assess the treatment's efficacy. Such latency periods challenge the assumption of rapid symptom relief and emphasize the necessity of extended trials to validate the therapeutic potential of hormone-based interventions.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [07_MED_NaiveRAG_b3] Domain: MED

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Why might the lack of difference between the thyroxine treatment and placebo groups in the study suggest a latency period rather than other potential biases or explanations?

**Reference Answer:** The lack of difference between the thyroxine treatment and placebo groups in the study suggests a latency period because the therapeutic effects of thyroxine in asymptomatic subclinical hypothyroidism may take longer than the 3-week follow-up period to manifest, thus not showing any significant improvement during this short observation time.

**System Answer:** The lack of difference between the thyroxine treatment and placebo groups in the study suggests a latency period rather than other potential biases or explanations because the observed effect may not be immediately apparent due to the time required for the therapeutic intervention to manifest its full impact. This latency period could indicate that the treatment's benefits might become evident only after a longer period, suggesting that the initial results do not necessarily reflect the true efficacy of the treatment. Other potential biases or explanations, such as measurement errors or selection bias, would typically show inconsistencies across different measurements or groups, whereas a latency period would explain the consistent lack of difference observed in the study.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [08_MED_NexRoute_v01] Domain: MED

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Why might the lack of publicly subsidized insurance for long-term care and personal assistance services pose a significant challenge in meeting the needs of adults with disabilities?

**Reference Answer:** The lack of publicly subsidized insurance for long-term care and personal assistance services poses a significant challenge because it leaves a substantial portion of the population without adequate financial support for essential services. This gap can lead to insufficient access to necessary care and services, which are critical for the well-being and independence of adults with disabilities. Additionally, the absence of such insurance may deter individuals from seeking needed support due to concerns about affordability, thereby exacerbating their dependency on informal care networks or other less effective alternatives.

**System Answer:** The lack of publicly subsidized insurance for long-term care and personal assistance services poses a significant challenge in meeting the needs of adults with disabilities because it leaves many individuals without adequate financial support to access necessary care and services. This gap in coverage can lead to unaffordable costs for individuals and their families, limiting their ability to obtain the support needed for daily living and potentially exacerbating health issues. Publicly subsidized insurance could help bridge this gap by providing financial assistance, ensuring that more people have access to the care they need without facing insurmountable financial barriers.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [08_MED_NaiveRAG_b3] Domain: MED

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Why might the lack of publicly subsidized insurance for long-term care and personal assistance services pose a significant challenge in meeting the needs of adults with disabilities?

**Reference Answer:** The lack of publicly subsidized insurance for long-term care and personal assistance services poses a significant challenge because it leaves a substantial portion of the population without adequate financial support for essential services. This gap can lead to insufficient access to necessary care and services, which are critical for the well-being and independence of adults with disabilities. Additionally, the absence of such insurance may deter individuals from seeking needed support due to concerns about affordability, thereby exacerbating their dependency on informal care networks or other less effective alternatives.

**System Answer:** The lack of publicly subsidized insurance for long-term care and personal assistance services poses a significant challenge because it leaves many individuals with disabilities without adequate financial support to cover the costs of necessary care and assistance. This can lead to inadequate care, increased stress on families, and a higher burden on the informal caregiving network. From a sociological perspective, it also highlights the need for more inclusive policies that recognize the diverse needs of aging populations and those with disabilities, ensuring they have access to the necessary support services.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [09_MED_NexRoute_v01] Domain: MED

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the pathophysiology and clinical presentation of right ventricular outflow obstruction with those of hypoplasia of the left ventricle to explain why right ventricular outflow obstruction is the most likely cause of the patient's symptoms.

**Reference Answer:** Right ventricular outflow obstruction is the most likely cause of the patient's symptoms due to its pathophysiology involving a narrowed pulmonary valve or infundibulum leading to increased right ventricular pressure and volume overload, which can result in cyanosis and poor weight gain. In contrast, hypoplasia of the left ventricle leads to inadequate left ventricular output and systemic hypoperfusion, typically presenting with signs of heart failure rather than the specific findings of a harsh systolic murmur and cyanosis during feeding and crying episodes.

**System Answer:** The pathophysiology of right ventricular outflow obstruction involves a narrowing or blockage at the level of the aortic valve or infundibulum, leading to increased afterload and subsequent left ventricular hypertrophy and dilation over time if left untreated. In contrast, hypoplasia of the left ventricle refers to underdeveloped or undergrown left ventricle, often resulting from congenital anomalies where the ventricle fails to fully form during fetal development, leading to reduced contractile function and volume output. Given the patient presents with signs of increased left ventricular workload such as hypertension and echocardiographic findings of left ventricular hypertrophy, along with the absence of clear evidence of left ventricle underdevelopment, it suggests an acquired mechanism of outflow obstruction rather than a primary structural defect in the left ventricle itself, making left ventricular outflow obstruction the more likely explanation for the patient's symptoms.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [09_MED_NaiveRAG_b3] Domain: MED

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the pathophysiology and clinical presentation of right ventricular outflow obstruction with those of hypoplasia of the left ventricle to explain why right ventricular outflow obstruction is the most likely cause of the patient's symptoms.

**Reference Answer:** Right ventricular outflow obstruction is the most likely cause of the patient's symptoms due to its pathophysiology involving a narrowed pulmonary valve or infundibulum leading to increased right ventricular pressure and volume overload, which can result in cyanosis and poor weight gain. In contrast, hypoplasia of the left ventricle leads to inadequate left ventricular output and systemic hypoperfusion, typically presenting with signs of heart failure rather than the specific findings of a harsh systolic murmur and cyanosis during feeding and crying episodes.

**System Answer:** The pathophysiology of right ventricular outflow obstruction involves a narrowing of the outflow tract, leading to increased right ventricular pressure and volume, while hypoplasia of the left ventricle results in reduced contractile function and chamber size. Clinically, patients with outflow obstruction often present with symptoms of right strain, such as syncope or dyspnea, whereas those with hypoplasia may have signs findings but no overt symptoms. The patient's history of syncope and family history of sudden death strongly suggest a mechanical obstruction rather than a functional defect, making outflow obstruction the more likely diagnosis.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [10_MED_NexRoute_v01] Domain: MED

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the etiology of stoma formation between carcinoma and other potential causes such as irritable bowel syndrome, hemorrhoids, and bladder instability, highlighting the relative frequency and significance of each condition.

**Reference Answer:** Carcinoma is one of the most common reasons for stoma formation compared to irritable bowel syndrome, hemorrhoids, and bladder instability, which are less frequently associated with this procedure.

**System Answer:** The etiology of stoma formation between carcinoma and other potential causes such as irritable bowel syndrome, hemorrhoids, and bladder instability highlights significant differences in both the relative frequency and the specific mechanisms involved. Carcinoma, particularly colorectal cancer, stands out as the most common reason for stoma formation due to its it accounts for a substantial proportion of all gastrointestinal malignancies globally, with approximately 1.3 million new cases annually. In this context, the creation of an ostomy provides a critical therapeutic option to manage cancer while it necessitates surgical intervention, thereby diverting waste products away from the affected area to prevent complications such as obstruction or infection, thus preserving both the integrity of the oncological treatment and the patient's quality of life. In contrast, conditions like irritable bowel syndrome, hemorrhoids, and bladder instability, although they can lead to symptoms requiring surgical intervention, including the formation of an ostomy, their incidence is significantly lower compared to carcinoma, making they often represent secondary rather than primary causes of stoma formation.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [10_MED_NaiveRAG_b3] Domain: MED

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the etiology of stoma formation between carcinoma and other potential causes such as irritable bowel syndrome, hemorrhoids, and bladder instability, highlighting the relative frequency and significance of each condition.

**Reference Answer:** Carcinoma is one of the most common reasons for stoma formation compared to irritable bowel syndrome, hemorrhoids, and bladder instability, which are less frequently associated with this procedure.

**System Answer:** The etiology of stoma formation in carcinoma is primarily due to the invasive nature of the tumor, leading to obstruction or damage to the intestines. This results in a higher frequency of stoma formation compared to other conditions like irritable bowel syndrome (IBS), hemorrhoids, and bladder instability, which have lower incidences of necessitating stoma creation. Carcinoma accounts for approximately 50% of stomas formed, while IBS contributes to about 2%, hemorrhoids contribute to around 3%, and bladder instability contributes to less than 1%.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [11_SOC_NexRoute_v01] Domain: SOC

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** What country saw its economic liberalization program interrupted by the deployment of 600,000 Soviet troops in 1968?

**Reference Answer:** Czechoslovakia

**System Answer:** The country that saw its economic liberalization program interrupted by the deployment of 600,000 Soviet troops in 1968 was Czechoslovakia. This event, known as the Warsaw Pact invasion of Czechoslovakia, occurred in August 1968 when the Soviet Union, along with other Eastern Bloc countries, intervened militarily to suppress the liberalizing reforms initiated by Alexander Dubcek's government, which aimed at increasing political freedom and democratization within the socialist framework, actions that were perceived as a threat to Soviet control and the broader communist bloc's stability, thus leading to the suspension of Czechoslovakia's economic liberalization initiatives and the reinstatement of more stringent controls over domestic politics and media.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [11_SOC_NaiveRAG_b3] Domain: SOC

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** What country saw its economic liberalization program interrupted by the deployment of 600,000 Soviet troops in 1968?

**Reference Answer:** Czechoslovakia

**System Answer:** Czechoslovakia

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [12_SOC_NexRoute_v01] Domain: SOC

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** How might individuals perceive the mechanism through which taking drugs can alleviate pain and lead to its use as a coping strategy?

**Reference Answer:** Individuals might perceive taking drugs as a mechanism to alleviate pain and discomfort, leading to its use as a coping strategy when dealing with physical or emotional pain, stress, or anxiety.

**System Answer:** Individuals may perceive the mechanism through which taking drugs alleviates pain and leads to its use as a coping strategy as a direct and immediate solution to emotional or physical discomfort, viewing the chemical alteration of their brain's pain pathways and mood regulation centers as a powerful tool to quickly mitigate distressing feelings or physical sensations. This perception can foster a reliance on substances not just for their analgesicic effects but also for their ability to provide a temporary escape from painful realities, thus reinforcing the cycle as a primary method for managing difficult emotions and situations without exploring other more sustainable or less harmful alternatives.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [12_SOC_NaiveRAG_b3] Domain: SOC

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** How might individuals perceive the mechanism through which taking drugs can alleviate pain and lead to its use as a coping strategy?

**Reference Answer:** Individuals might perceive taking drugs as a mechanism to alleviate pain and discomfort, leading to its use as a coping strategy when dealing with physical or emotional pain, stress, or anxiety.

**System Answer:** Individuals may perceive the mechanism through which drugs can alleviate pain as a direct intervention against the source of their discomfort, leading them to view drug use as a legitimate coping strategy. This perception can be influenced by both the pharmacological effects of the drugs and the social and cultural attitudes towards pain management.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [13_SOC_NexRoute_v01] Domain: SOC

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the concept of using cultural institutions to control and strengthen cultural values with the use of military force to expel social groups from a state, and explain the fundamental differences in their approaches and implications.

**Reference Answer:** While both methods aim at influencing social dynamics within a state, using cultural institutions to control and strengthen cultural values (such as through churches and schools) operates through soft power and long-term socialization, aiming to perpetuate cultural norms and values over generations. In contrast, engaging military forces to expel certain social groups is a coercive and immediate measure that involves direct physical displacement and can lead to significant human rights violations and social upheaval.

**System Answer:** The concept of using cultural institutions to control and strengthen cultural values involves leveraging education systems, media, and other cultural venues to shape public opinion and reinforce societal norms, aiming to promote unity and shared values within a community. In contrast, using military force to expel social groups from a state represents an extreme and coercive measure, often associated with ethnic cleansing or forced migration, which can lead to significant social division, human rights violations, and long-term instability. The former seeks to foster social cohesion and integration, while the latter disrupts social structures and can perpetuate cycles of conflict and mistrust.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [13_SOC_NaiveRAG_b3] Domain: SOC

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the concept of using cultural institutions to control and strengthen cultural values with the use of military force to expel social groups from a state, and explain the fundamental differences in their approaches and implications.

**Reference Answer:** While both methods aim at influencing social dynamics within a state, using cultural institutions to control and strengthen cultural values (such as through churches and schools) operates through soft power and long-term socialization, aiming to perpetuate cultural norms and values over generations. In contrast, engaging military forces to expel certain social groups is a coercive and immediate measure that involves direct physical displacement and can lead to significant human rights violations and social upheaval.

**System Answer:** Both using cultural institutions to control and strengthen cultural values and employing military force to expel social groups share the goal of maintaining social order and controlling population dynamics. However, they differ fundamentally in their methods and ethical implications. Cultural institutions typically aim to influence behavior through education, norms, and traditions, fostering a sense of belonging and shared identity. In contrast, military force involves direct intervention and coercion, raising significant ethical concerns about human rights and social justice. The former approach is generally seen as less invasive and more sustainable, while the latter can lead to severe social unrest and violate basic human rights.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [14_SOC_NexRoute_v01] Domain: SOC

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the availability and typical pricing of grapes in a food store versus a wine country setting, considering factors such as supply chain, direct sourcing, and market dynamics.

**Reference Answer:** While both food stores and wine country offer grapes, the availability and pricing differ significantly. Food stores provide a consistent supply of grapes throughout the year from various global sources, which can affect prices based on market demand and supply. In contrast, wine country often sells local, seasonally available grapes directly from vineyards or small-scale producers, potentially at higher prices due to the direct-to-consumer model and the added value of local produce. The choice between the two depends on the consumer's preference for convenience, price, and the perceived quality and freshness of the product.

**System Answer:** In a food store setting, grapes are typically sourced through established supply chains involving distributors and wholesalers, which can lead to standardized pricing based on market demand and competition. The availability of grapes in food stores is usually consistent throughout the year, with some varieties being more abundant during certain seasons. In contrast, a wine country setting often involves direct sourcing from local vineyards, where grapes may be sold directly to winemakers or through local markets, potentially leading to more variable pricing depending on the season and harvest. The direct relationship between growers and buyers in a wine country setting can result in fresher produce and sometimes higher prices due to the premium nature of the product. However, this also means that consumers in a wine country might benefit from locally grown grapes that are better suited for specific regional wines, enhancing both the quality and authenticity of the final product.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [14_SOC_NaiveRAG_b3] Domain: SOC

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the availability and typical pricing of grapes in a food store versus a wine country setting, considering factors such as supply chain, direct sourcing, and market dynamics.

**Reference Answer:** While both food stores and wine country offer grapes, the availability and pricing differ significantly. Food stores provide a consistent supply of grapes throughout the year from various global sources, which can affect prices based on market demand and supply. In contrast, wine country often sells local, seasonally available grapes directly from vineyards or small-scale producers, potentially at higher prices due to the direct-to-consumer model and the added value of local produce. The choice between the two depends on the consumer's preference for convenience, price, and the perceived quality and freshness of the product.

**System Answer:** In a food store, grapes are likely sourced through conventional supply chains involving multiple intermediaries, which can affect the final price due to transportation costs and markups. Direct sourcing from a wine country setting would eliminate some of these middlemen, potentially resulting in lower prices for consumers while still allowing producers to maintain reasonable margins. Market dynamics in wine regions might also favor local growers who produce both table grapes and wine, leading to more integrated farming practices and possibly better quality at similar or lower prices compared to non-wine regions.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [15_SOC_NexRoute_v01] Domain: SOC

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** What aspect of the classroom environment might make it less interesting to children compared to other locations such as a toy store, school, or soccer game?

**Reference Answer:** The structured and formal nature of the classroom environment, along with its limited opportunities for hands-on exploration and physical activity, might make it less interesting to children compared to the more engaging and interactive settings found in a toy store, school (which can offer various activities), or a soccer game (which involves physical play and social interaction).

**System Answer:** Classrooms often fail to capture the interest of children due to their rigid structure and limited interactive elements, which are crucial for engaging young minds according to social science research on child development and learning environments. Unlike toy stores where children can freely explore and interact with a wide variety of stimulating objects, schools settings that might include incorporate more hands-on activities and peer interaction, or soccer games which offer offer dynamic physical engagement and immediate feedback, classrooms tend to be more structured around did lecturing-based instruction and passive learning, thereby failing to meet the developmental needs of young children who thrive in spaces that offer them the freedom to explore, learn through play, and interact with both peers and materials in a hands-on manner.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [15_SOC_NaiveRAG_b3] Domain: SOC

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** What aspect of the classroom environment might make it less interesting to children compared to other locations such as a toy store, school, or soccer game?

**Reference Answer:** The structured and formal nature of the classroom environment, along with its limited opportunities for hands-on exploration and physical activity, might make it less interesting to children compared to the more engaging and interactive settings found in a toy store, school (which can offer various activities), or a soccer game (which involves physical play and social interaction).

**System Answer:** The aspect of the classroom environment that might make it less interesting to children compared to other locations such as a toy store, school, or soccer game is the lack of engaging and stimulating activities tailored to their age and interests. Classrooms often lack the variety and excitement found in places like toy stores or soccer games, which can make them less appealing to children.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [16_STEM_NexRoute_v01] Domain: STEM

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Why can bacterial sexually transmitted infections (STIs) be cured with antibiotics while viral STIs cannot?

**Reference Answer:** Bacterial sexually transmitted infections (STIs) can be cured with antibiotics because antibiotics target and eliminate bacterial pathogens directly. In contrast, viral STIs cannot be cured with antibiotics because viruses are not affected by these medications; they specifically target and destroy bacterial cells, which makes them ineffective against viral infections. For viral STIs, the body's immune system must work to manage and control the infection, often leading to a chronic condition where the virus remains in the body even after symptoms subside.

**System Answer:** Bacterial sexually transmitted infections (STIs) can be cured with antibiotics because bacteria are living organisms that can be targeted and eliminated by antibiotics, which work by either disrupting cell walls, inhibiting protein synthesis, or interfering with metabolic processes essential for bacterial survival and reproduction. In contrast, viral STIs cannot be cured with antibiotics because viruses are not living organisms but rather genetic material that hijacks host the host cells' machinery to replicate, making antibiotics ineffective against them; instead, antiviral medications must target specific aspects of the viral life cycle, such as its replication mechanisms or entry into cells, which requires more complex and often times less effective treatments compared to those for bacterial infections.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [16_STEM_NaiveRAG_b3] Domain: STEM

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Why can bacterial sexually transmitted infections (STIs) be cured with antibiotics while viral STIs cannot?

**Reference Answer:** Bacterial sexually transmitted infections (STIs) can be cured with antibiotics because antibiotics target and eliminate bacterial pathogens directly. In contrast, viral STIs cannot be cured with antibiotics because viruses are not affected by these medications; they specifically target and destroy bacterial cells, which makes them ineffective against viral infections. For viral STIs, the body's immune system must work to manage and control the infection, often leading to a chronic condition where the virus remains in the body even after symptoms subside.

**System Answer:** Bacterial sexually transmitted infections (STIs) can be cured with antibiotics because they target and eliminate bacteria directly. In contrast, viral STIs cannot be cured with antibiotics because antibiotics do not affect viruses; instead, antiviral medications or other therapeutic approaches are required to manage symptoms and prevent complications.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [17_STEM_NexRoute_v01] Domain: STEM

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the visibility of stars in the night sky when the moon is full versus when it is new, explaining the underlying astronomical reasons for the difference.

**Reference Answer:** Stars appear more visible in the night sky when the moon is new compared to when it is full due to the reduced light pollution from the moon during a new moon, allowing fainter stars to be seen. During a full moon, the bright lunar illumination can obscure the less luminous stars, making them harder to observe.

**System Answer:** When the moon is full, it significantly outshines other stars in the night sky due to its position relative to the sun and Earth, reflecting nearly all of the sun's light directly back towards us, creating an intense illumination that d covers a large portion of the sky, thereby making it much harder to see other fainter stars against the bright lunar backdrop. Conversely, during a new moon, when the moon is between the sun and Earth, very little of its illuminated side faces us, meaning that it appears almost completely dark from our vantage point, allowing for clear visibility of many more stars and other other celestial objects that might otherwise be lost in the glare of the full moon's light.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [17_STEM_NaiveRAG_b3] Domain: STEM

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the visibility of stars in the night sky when the moon is full versus when it is new, explaining the underlying astronomical reasons for the difference.

**Reference Answer:** Stars appear more visible in the night sky when the moon is new compared to when it is full due to the reduced light pollution from the moon during a new moon, allowing fainter stars to be seen. During a full moon, the bright lunar illumination can obscure the less luminous stars, making them harder to observe.

**System Answer:** The visibility of stars in the night sky is significantly impacted by the phase of the moon. When the moon is full, its bright light can obscure many stars, making them less visible. Conversely, during a new moon, there is no direct illumination from the sun to the moon, allowing more stars to be seen. This difference is due to the varying positions of the Earth, Moon, and Sun in relation to each other throughout the lunar cycle.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [18_STEM_NexRoute_v01] Domain: STEM

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the genetic mechanism of mutation with the concept of epigenetic modification in terms of their effects on gene expression and heritability.

**Reference Answer:** Both mutation and epigenetic modification can affect gene expression, but they do so through different mechanisms. Mutation involves a change in the DNA sequence, which can be heritable and lead to permanent changes in the genetic makeup of an organism. In contrast, epigenetic modifications involve chemical changes to DNA or associated proteins that do not alter the DNA sequence itself, and these changes can be reversible and not always heritable, influencing gene expression without changing the underlying DNA sequence.

**System Answer:** Both mutation and epigenetic modifications represent fundamental mechanisms that influence gene expression and contribute to phenotypic diversity, yet they operate at distinct levels of genetic information. Mutations involve alterations in the DNA sequence itself, such as insert nucleotide substitutions, insertions, deletions, or translocations, which directly change the genetic blueprint passed on to subsequent generations, thereby altering the potential for heritability. In contrast, epigenetic modifications do not change the DNA sequence but instead modify how genes expression is regulated without chemical tags like methylation or histone modifications. These changes do not alter the genetic code per se but can be stably maintained across cell divisions and even through meiosis, allowing for heritable changes in gene expression without affecting the actual sequence of DNA. Thus, while both mechanisms can lead to heritable phenotypic differences, the former does so through permanent alterations in the genetic sequence, while the latter does so through reversible changes in gene regulation, highlighting the complex interplay between genetic and epigenetic factors in shaping individual traits and susceptibility to diseases.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [18_STEM_NaiveRAG_b3] Domain: STEM

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the genetic mechanism of mutation with the concept of epigenetic modification in terms of their effects on gene expression and heritability.

**Reference Answer:** Both mutation and epigenetic modification can affect gene expression, but they do so through different mechanisms. Mutation involves a change in the DNA sequence, which can be heritable and lead to permanent changes in the genetic makeup of an organism. In contrast, epigenetic modifications involve chemical changes to DNA or associated proteins that do not alter the DNA sequence itself, and these changes can be reversible and not always heritable, influencing gene expression without changing the underlying DNA sequence.

**System Answer:** Both genetic mutations and epigenetic modifications can alter gene expression, but they do so through different mechanisms. Genetic mutations involve changes in the DNA sequence itself, which can lead to permanent alterations in the gene's function and can be passed on to offspring, thus having a heritable effect. In contrast, epigenetic modifications, such as DNA methylation or histone modifications, do not change the DNA sequence; instead, they regulate gene expression by adding chemical tags to the DNA or histones without altering the underlying sequence. These changes are generally not heritable across generations unless they occur in germ cells. Both mechanisms play crucial roles in developmental processes and can be influenced by environmental factors, highlighting the complex interplay between genetics and environment in shaping phenotypes.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [19_STEM_NexRoute_v01] Domain: STEM

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the effects of ice wedging caused by the freezing and thawing of H2O versus those caused by the freezing and thawing of water in plant tissues on rock surfaces.

**Reference Answer:** Ice wedging caused by the freezing and thawing of H2O has a more significant effect on rock surfaces compared to the freezing and thawing of water in plant tissues. The process of ice wedging due to H2O involves the expansion of water as it freezes, creating pressure that can split rocks. In contrast, the freezing and thawing of water in plant tissues primarily affects the plants themselves rather than directly causing physical damage to rocks, though it can influence soil structure and hydrology indirectly.

**System Answer:** Ice wedging caused by the freezing and thawing of water in rocks or soil differs significantly from that caused by water in plant tissues due to their distinct environmental contexts and mechanisms of action. In In rocks or soil, water enters pre-existing cracks and crevices where it freezes and expands, exerting pressure that gradually widens these fissures, potentially leading to the disintegration of larger rock formations. Conversely, in plant tissues, water freezing directly damages cells by causing ice crystal formation within them, leading to cell wall rupture and tissue death, although this process does not physically alter the surrounding environment but rather affects the plant itself. Therefore, while both processes involve the expansion of water upon freezing, their impacts vary greatly between altering rock structures versus destroying plant tissues.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [19_STEM_NaiveRAG_b3] Domain: STEM

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the effects of ice wedging caused by the freezing and thawing of H2O versus those caused by the freezing and thawing of water in plant tissues on rock surfaces.

**Reference Answer:** Ice wedging caused by the freezing and thawing of H2O has a more significant effect on rock surfaces compared to the freezing and thawing of water in plant tissues. The process of ice wedging due to H2O involves the expansion of water as it freezes, creating pressure that can split rocks. In contrast, the freezing and thawing of water in plant tissues primarily affects the plants themselves rather than directly causing physical damage to rocks, though it can influence soil structure and hydrology indirectly.

**System Answer:** Both ice wedging processes involve the expansion of water upon freezing and the contraction upon thawing, leading to mechanical stress on the surface material. However, the specific effects differ based on the material. For rocks, ice wedging can cause significant fragmentation and erosion over time due to the sheer force exerted by the expanding ice crystals. In contrast, in plant tissues, the process of ice wedging can lead to the rupture of cell walls and the formation of ice lenses within the tissue, potentially causing structural damage to the plant but not necessarily leading to the same level of physical degradation seen in rocks. The key difference lies in the scale and nature of the damage: rocks experience mechanical breakdown, while plants face cellular-level disruption.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [20_STEM_NexRoute_v01] Domain: STEM

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** What characteristic distinguishes warm-blooded animals from cold-blooded ones, and which of these creatures, the toad, snake, turtle, or skunk, exemplifies this characteristic?

**Reference Answer:** Warm-blooded animals maintain a constant body temperature through internal mechanisms, regardless of the external environment. The skunk exemplifies this characteristic among the given options.

**System Answer:** Warm-blooded animals, also known as endotherms, distinguish themselves from cold-blooded animals, or ectotherms, by their ability to maintain a stable internal body temperature through internal metabolic processes, regardless of the external environment. This characteristic is exemplified by the skunk among the given options; it regulates its body temperature internally through metabolic adjustments such as shivering and adjusting metabolic rates, allowing it to be active in various temperatures and maintain consistent bodily functions, thereby distinguishing it from cold-blooded animals which rely on external sources to regulate their body temperature.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [20_STEM_NaiveRAG_b3] Domain: STEM

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** What characteristic distinguishes warm-blooded animals from cold-blooded ones, and which of these creatures, the toad, snake, turtle, or skunk, exemplifies this characteristic?

**Reference Answer:** Warm-blooded animals maintain a constant body temperature through internal mechanisms, regardless of the external environment. The skunk exemplifies this characteristic among the given options.

**System Answer:** Warm-blooded animals maintain a relatively constant body temperature regardless of the environment, whereas cold-blooded animals rely on external sources to regulate their body temperature. Among the given options, the skunk is an example of a warm-blooded animal, as it maintains a stable body temperature through internal physiological mechanisms.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [21_CROSS_NexRoute_v01] Domain: CROSS

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** What does Socrates' assertion that 'the unexamined life is not worth living' imply about the relationship between individual introspection and societal norms, especially in the context of developing personal ethics based on scientific understanding?

**Reference Answer:** Socrates' assertion emphasizes the importance of individual introspection over blind adherence to societal norms, suggesting that a person's ethical framework should be grounded in thoughtful consideration rather than conformity. This implies that scientific understanding and personal ethics should be intertwined, where individuals critically examine their actions and beliefs to foster a more rational and morally sound society.

**System Answer:** Socrates' assertion that 'the unexamined life is not worth living' implies that individual introspection and self-examination are crucial for developing a coherent set of personal ethics. This idea resonates with the scientific understanding of human cognition and moral development, suggesting that engaging in critical thinking and reflective practice allows individuals to align their actions with rational principles and societal expectations, fostering a more harmonious and meaningful existence within community structures.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [21_CROSS_NaiveRAG_b3] Domain: CROSS

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** What does Socrates' assertion that 'the unexamined life is not worth living' imply about the relationship between individual introspection and societal norms, especially in the context of developing personal ethics based on scientific understanding?

**Reference Answer:** Socrates' assertion emphasizes the importance of individual introspection over blind adherence to societal norms, suggesting that a person's ethical framework should be grounded in thoughtful consideration rather than conformity. This implies that scientific understanding and personal ethics should be intertwined, where individuals critically examine their actions and beliefs to foster a more rational and morally sound society.

**System Answer:** Socrates' assertion emphasizes the importance of individual introspection and critical thinking over blind adherence to societal norms when developing personal ethics. It suggests that personal ethics should be grounded in rational reflection rather than external pressures or technological influences, ensuring that individuals make informed and meaningful choices aligned with their inherent human virtues.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [22_CROSS_NexRoute_v01] Domain: CROSS

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the concept of an anointed leader in religious contexts with the role of a CEO in modern business organizations, considering their origins, responsibilities, and the mechanisms through which they gain authority and influence.

**Reference Answer:** Both the concept of an anointed leader in religious contexts and the role of a CEO in modern business organizations share historical roots of leadership and authority. In religious traditions, such as Judaism, the term 'mashiach' refers to the anointed one, symbolizing a leader chosen by divine will, often associated with messianic expectations. Similarly, CEOs in business are often seen as leaders chosen by stakeholders to guide and manage organizations effectively. However, while the anointed leader's authority is rooted in spiritual and divine endorsement, a CEO's authority is based on contractual agreements and performance in the business environment. Both roles involve gaining and maintaining influence through various mechanisms, such as vision setting, decision-making, and interpersonal relationships, but operate within distinct frameworks of governance and accountability.

**System Answer:** An anointed leader in religious contexts and a CEO in modern business organizations share several conceptual parallels, including the idea of being chosen or appointed for a specific role, having significant influence over others, and wielding authority within their respective domains. However, there are also distinct differences. An anointed leader often derives their power from spiritual or divine sources, whereas a CEO’s authority stems from formal organizational structures and legal frameworks. Both roles involve leading and guiding their followers or employees toward specific goals, but the methods and means of achieving these ends differ significantly, with religious leaders focusing on moral and spiritual guidance, while CEOs concentrate on strategic planning and operational efficiency.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [22_CROSS_NaiveRAG_b3] Domain: CROSS

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the concept of an anointed leader in religious contexts with the role of a CEO in modern business organizations, considering their origins, responsibilities, and the mechanisms through which they gain authority and influence.

**Reference Answer:** Both the concept of an anointed leader in religious contexts and the role of a CEO in modern business organizations share historical roots of leadership and authority. In religious traditions, such as Judaism, the term 'mashiach' refers to the anointed one, symbolizing a leader chosen by divine will, often associated with messianic expectations. Similarly, CEOs in business are often seen as leaders chosen by stakeholders to guide and manage organizations effectively. However, while the anointed leader's authority is rooted in spiritual and divine endorsement, a CEO's authority is based on contractual agreements and performance in the business environment. Both roles involve gaining and maintaining influence through various mechanisms, such as vision setting, decision-making, and interpersonal relationships, but operate within distinct frameworks of governance and accountability.

**System Answer:** An anointed leader in religious contexts typically originates from divine selection or prophetic calling, focusing on spiritual guidance and moral leadership. They gain authority through faith and community endorsement, often serving as a symbol of hope and unity. In contrast, a CEO in modern business organizations arises from professional qualifications and market performance, emphasizing strategic decision-making and operational efficiency. Both roles involve gaining authority and influence but operate within different frameworks: one rooted in spirituality and tradition, the other in secular governance and economic dynamics.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [23_CROSS_NexRoute_v01] Domain: CROSS

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the geopolitical implications of the Truman Doctrine as a legal and scientific policy instrument, discussing how it influenced international law and the emerging field of international relations studies.

**Reference Answer:** The Truman Doctrine had significant legal implications by establishing a precedent for US intervention in foreign conflicts under the guise of containment against communism, which influenced subsequent international legal frameworks such as the United Nations Charter and the laws of armed conflict. Scientifically, it marked the beginning of a new era in international relations studies, shifting the focus from traditional balance-of-power theories to a more geographically and ideologically driven approach, which has been studied extensively in fields like geopolitics and international security studies.

**System Answer:** The geopolitical implications of the Truman Doctrine as a legal and scientific policy instrument are multifacdimensional. On one hand, from a legal perspective, the Truman Doctrine significantly shaped the development of international law by establishing precedents for interventionist policies and the concept of containment, which became foundational in post-war international relations. It also set a precedent for the use of economic aid as a tool for influencing foreign policy, thereby influencing subsequent legal frameworks and treaties. From a scientific standpoint, particularly in the realm of international relations studies, the Truman Doctrine contributed to the emergence of behavioral science approaches in diplomacy and security studies. It provided empirical data and case studies that researchers could analyze to understand the efficacy of different foreign policies and the role of ideology in shaping international behavior. Both domains saw the doctrine as a critical turning point, with legal scholars and social scientists alike examining its long-term effects on global governance structures and theoretical frameworks.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [23_CROSS_NaiveRAG_b3] Domain: CROSS

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Compare the geopolitical implications of the Truman Doctrine as a legal and scientific policy instrument, discussing how it influenced international law and the emerging field of international relations studies.

**Reference Answer:** The Truman Doctrine had significant legal implications by establishing a precedent for US intervention in foreign conflicts under the guise of containment against communism, which influenced subsequent international legal frameworks such as the United Nations Charter and the laws of armed conflict. Scientifically, it marked the beginning of a new era in international relations studies, shifting the focus from traditional balance-of-power theories to a more geographically and ideologically driven approach, which has been studied extensively in fields like geopolitics and international security studies.

**System Answer:** The Truman Doctrine had significant geopolitical implications as both a legal and scientific policy instrument. Legally, it established a precedent for the United States to intervene in foreign affairs under the guise of national security, which was codified in various international agreements and laws. Scientifically, the doctrine contributed to the development of new fields in international relations studies, such as the study of containment and the role of soft power in diplomacy. Its influence extended beyond legal and scientific realms, shaping the broader narrative of the Cold War and contributing to the formation of the modern discipline of international relations.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [24_CROSS_NexRoute_v01] Domain: CROSS

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Why did peace researchers in the 1970s face significant challenges in broadening the scope of peace studies despite the increasing global focus on conflict prevention and peacekeeping efforts?

**Reference Answer:** Peace researchers in the 1970s faced significant challenges in broadening the scope of peace studies because the field was often viewed skeptically during the height of the Cold War, leading to criticism that it was 'appeasement studies,' and there was a strong opposition from critics who saw it as undermining national security concerns.

**System Answer:** Peace researchers in the 1990s faced significant challenges in broadening the scope of peace studies despite the increasing global focus on conflict prevention and peacekeeping efforts because the field remained nascent and lacked established institutional support, including specialized journals and formal academic programs, which hindered the dissemination of new ideas and the integration of interdisciplinary approaches. This nascent stage meant that peace studies were still struggling to gain recognition as a legitimate academic discipline, making its practitioners often had to operate outside traditional scholarly channels, leading to difficulties in gaining traction and influence within the broader academic community and policy-making circles.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [24_CROSS_NaiveRAG_b3] Domain: CROSS

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Why did peace researchers in the 1970s face significant challenges in broadening the scope of peace studies despite the increasing global focus on conflict prevention and peacekeeping efforts?

**Reference Answer:** Peace researchers in the 1970s faced significant challenges in broadening the scope of peace studies because the field was often viewed skeptically during the height of the Cold War, leading to criticism that it was 'appeasement studies,' and there was a strong opposition from critics who saw it as undermining national security concerns.

**System Answer:** Peace researchers in the 1970s faced significant challenges in broadening the scope of peace studies due to several factors. Firstly, the focus on conflict prevention and peacekeeping efforts often reinforced existing power dynamics and Western-centric views, making it difficult to integrate diverse perspectives and actors. Additionally, the reliance on rationalist foundations within constructivism limited the applicability of these theories in a rapidly changing international system. Furthermore, the increasing relevance of security studies in addressing societal concerns sometimes overshadowed the need for a more holistic approach, including contributions from mathematicians and natural scientists. These challenges highlight the need for a more inclusive and interdisciplinary approach to peace studies.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [25_CROSS_NexRoute_v01] Domain: CROSS

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Why did the Multicenter Randomized CLinical trial of Endovascular treatment for Acute ischemic stroke in the Netherlands (MR CLEAN) demonstrate better outcomes with endovascular treatment compared to medical therapy, and how does this finding reflect broader sociocultural factors influencing healthcare practices?

**Reference Answer:** The MR CLEAN study demonstrated better outcomes with endovascular treatment compared to medical therapy for acute ischemic stroke due to the rapid revascularization achieved through mechanical thrombectomy, which improves blood flow to the brain more effectively than standard medical management alone. Socioculturally, this finding reflects a shift towards more aggressive and technology-driven approaches in stroke care, driven by advancements in medical science and the increasing emphasis on reducing disability and improving patient outcomes in society.

**System Answer:** The MR CLEAN trial showed better outcomes with endovascular treatment due to its rigorous design and large sample size, which allowed for more reliable statistical analysis. Socioculturally, these findings reflect a shift towards more aggressive and technologically advanced treatments in acute stroke management, driven by advancements in medical technology and the desire to improve patient outcomes quickly. However, they also highlight the importance of cultural acceptance and integration of new technologies into clinical practice, as well as the potential for varying levels of adoption across different healthcare systems.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```


## [25_CROSS_NaiveRAG_b3] Domain: CROSS

```
You are an expert evaluator for question-answering systems.
Rate the following AI-generated answer on three criteria (1–5 integer scale each):

  1. **Factual Accuracy** — Is the answer factually correct and free of contradictions?
     1=Clearly wrong/contradictory, 3=Partially correct, 5=Fully accurate
  2. **Relevance** — Does the answer directly address the question?
     1=Off-topic, 3=Partially relevant, 5=Fully on-topic
  3. **Coherence** — Is the answer well-structured and easy to follow?
     1=Incoherent, 3=Mostly clear, 5=Clear and well-organized

**Question:** Why did the Multicenter Randomized CLinical trial of Endovascular treatment for Acute ischemic stroke in the Netherlands (MR CLEAN) demonstrate better outcomes with endovascular treatment compared to medical therapy, and how does this finding reflect broader sociocultural factors influencing healthcare practices?

**Reference Answer:** The MR CLEAN study demonstrated better outcomes with endovascular treatment compared to medical therapy for acute ischemic stroke due to the rapid revascularization achieved through mechanical thrombectomy, which improves blood flow to the brain more effectively than standard medical management alone. Socioculturally, this finding reflects a shift towards more aggressive and technology-driven approaches in stroke care, driven by advancements in medical science and the increasing emphasis on reducing disability and improving patient outcomes in society.

**System Answer:** The MR CLEAN study demonstrated better outcomes with endovascular treatment compared to medical therapy due to the direct mechanical removal of occlusive material, which can restore blood flow more effectively than medical therapy alone. Socioculturally, this finding reflects the increasing emphasis on evidence-based medicine and the adoption of advanced technologies in healthcare practices, as well as the potential for policy changes that could lead to more widespread access to these treatments. The study's results may also influence public health priorities by highlighting the benefits of early intervention and the importance of specialized centers for treating acute ischemic stroke.

Respond with ONLY a JSON object, nothing else:
{"factual_accuracy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "reasoning": "<one sentence>"}

```

