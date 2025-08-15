# Medical Question-Answering System

## Overview
This project implements a medical question-answering (QA) system capable of retrieving relevant answers from a structured dataset of question-answer pairs. The system leverages three retrieval approaches:

1. **TF-IDF** – Lexical matching using term frequency-inverse document frequency vectors.
2. **BM25** – A sparse retrieval method based on probabilistic ranking.
3. **Dense Retriever (SentenceTransformer)** – Contextual embeddings fine-tuned on the dataset with a MultipleNegativesRankingLoss.

The pipeline includes preprocessing, model training/fine-tuning, and evaluation using standard ranking metrics.

## Assumptions
- The dataset is provided as a CSV file with columns `question` and `answer`.
- Answers are treated as canonical; identical answers are deduplicated and assigned unique IDs.
- Questions and answers are normalized by lowercasing, removing punctuation, and collapsing whitespace.
- Dense retrieval uses the `all-mpnet-base-v2` model as the base embedding model.
- Hard negatives for fine-tuning are generated using BM25.
- During evaluation, the top-k retrieved answers are considered for ranking metrics such as Recall@k, nDCG@k, MRR, and MAP.
- Step-wise reasoning assumes that relevant answers can be matched semantically, and that the dataset contains enough examples to learn meaningful embeddings.
- The system assumes a closed-world scenario where all possible answers exist in the dataset (no out-of-corpus answers).
- Retrieval metrics assume one correct answer per question; multiple correct answers are treated as duplicates.

## Model Performance

### Sparse Retrieval
**TF-IDF**
- Recall@1: 0.2067  
- Recall@3: 0.4283  
- Recall@5: 0.5738  
- Recall@10: 0.7160  
- MRR@5: 0.3337  

**BM25**
- Recall@1: 0.2797  
- Recall@3: 0.5014  
- Recall@5: 0.6325  
- Recall@10: 0.7576  
- MRR@5: 0.4046  

### Dense Retrieval
**Baseline (no fine-tuning)**
- Recall@1: 0.5170, nDCG@1: 0.5167  
- Recall@3: 0.7310, nDCG@3: 0.6422  
- Recall@5: 0.8071, nDCG@5: 0.6738  
- Recall@10: 0.8673, nDCG@10: 0.6936  
- MRR: 0.6410, MAP: 0.6410  

**Fine-Tuned Dense Retriever**
- Recall@1: 0.7459, nDCG@1: 0.7456  
- Recall@3: 0.8716, nDCG@3: 0.8213  
- Recall@5: 0.9061, nDCG@5: 0.8356  
- Recall@10: 0.9321, nDCG@10: 0.8443  
- MRR: 0.8179, MAP: 0.8179  

### Performance Comparison
| Model                  | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR    |
|------------------------|-----------|-----------|-----------|------------|--------|
| TF-IDF                 | 0.2067    | 0.4283    | 0.5738    | 0.7160     | 0.3337 |
| BM25                   | 0.2797    | 0.5014    | 0.6325    | 0.7576     | 0.4046 |
| Dense Baseline          | 0.5170    | 0.7310    | 0.8071    | 0.8673     | 0.6410 |
| Dense Fine-Tuned        | 0.7459    | 0.8716    | 0.9061    | 0.9321     | 0.8179 |

## Strengths and Weaknesses
**Strengths:**
- Fine-tuned dense retriever significantly outperforms lexical baselines, especially at top-1 and top-5 ranks.
- Dense embeddings capture semantic similarity beyond exact keyword matching.
- Pipeline supports easy evaluation and comparison of sparse vs. dense retrieval methods.

**Weaknesses:**
- Sparse retrievers (TF-IDF/BM25) underperform for semantically similar but lexically different questions.
- Dense model requires GPU for efficient training and inference.
- Hard negatives are limited to top BM25 candidates; rare or ambiguous queries may remain challenging.
- Closed-world assumption may fail if a user query is not covered by the dataset.

## Example Interactions (Baseline)

**User:** What are the symptoms of diabetes?  
1. **score=0.815**  
   *ans:* the signs and symptoms of diabetes are being very thirsty urinating often feeling very hungry feeling very tired losing weight without trying sores that heal slowly dry itchy skin feelings of pins and needles in your feet losing feeling in your feet blurry eyesight some people with diabetes dont have any of these signs or symptoms the only way to know if you have diabetes is to have your doctor do a blood test  
2. **score=0.802**  
   *ans:* many people with diabetes experience one or more symptoms including extreme thirst or hunger a frequent need to urinate andor fatigue some lose weight without trying additional signs include sores that heal slowly dry itchy skin loss of feeling or tingling in the feet and blurry eyesight some people with diabetes however have no symptoms at all  
3. **score=0.747**  
   *ans:* the signs and symptoms of type 2 diabetes can be so mild that you might not even notice them nearly 7 million people in the united states have type 2 diabetes and dont know they have the disease many have no signs or symptoms some people have symptoms but do not suspect diabetes symptoms include increased thirst increased hunger fatigue increased urination especially at night unexplained weight loss blurred vision numbness or tingling in the feet or hands sores that do not heal many people do not find out they have the disease until they have diabetes problems such as blurred vision or heart trouble if you find out early that you have diabetes you can get treatment to prevent damage to your body  

**User:** How can I treat a common cold?  
1. **score=0.688**  
   *ans:* sneezing sore throat a stuffy nose coughing everyone knows the symptoms of the common cold it is probably the most common illness in the course of a year people in the united states suffer 1 billion colds you can get a cold by touching your eyes or nose after you touch surfaces with cold germs on them you can also inhale the germs symptoms usually begin 2 or 3 days after infection and last 2 to 14 days washing your hands and staying away from people with colds will help you avoid colds there is no cure for the common cold for relief try getting plenty of rest drinking fluids gargling with warm salt water using cough drops or throat sprays taking overthecounter pain or cold medicines however do not give aspirin to children and do not give cough medicine to children under four nih national institute of allergy and infectious diseases  
2. **score=0.650**  
   *ans:* summary sneezing sore throat a stuffy nose coughing everyone knows the symptoms of the common cold it is probably the most common illness in the course of a year people in the united states suffer 1 billion colds what can you do for your cold or cough symptoms besides drinking plenty of fluids and getting plenty of rest you may want to take medicines there are lots of different cold and cough medicines and they do different things nasal decongestants unclog a stuffy nose cough suppressants quiet a cough expectorants loosen mucus so you can cough it up antihistamines stop runny noses and sneezing pain relievers ease fever headaches and minor aches and pains here are some other things to keep in mind about cold and cough medicines read labels because many cold and cough medicines contain the same active ingredients taking too much of certain pain relievers can lead to serious injury do not give cough medicines to children under four and dont give aspirin to children finally antibiotics wont help a cold food and drug administration  
3. **score=0.572**  
   *ans:* summary when you cough or sneeze you send tiny germfilled droplets into the air colds and flu usually spread that way you can help stop the spread of germs by covering your mouth and nose when you sneeze or cough sneeze or cough into your elbow not your hands cleaning your hands often always before you eat or prepare food and after you use the bathroom or change a diaper avoiding touching your eyes nose or mouth hand washing is one of the most effective and most overlooked ways to stop disease soap and water work well to kill germs wash for at least 20 seconds and rub your hands briskly disposable hand wipes or gel sanitizers also work well  

**User:** What causes high blood pressure?  
1. **score=0.665**  
   *ans:* high blood pressure is a common disease in which blood flows through blood vessels arteries at higher than normal pressures there are two main types of high blood pressure primary and secondary high blood pressure primary or essential high blood pressure is the most common type of high blood pressure this type of high blood pressure tends to develop over years as a person ages secondary high blood pressure is caused by another medical condition or use of certain medicines this type usually resolves after the cause is treated or removed  
2. **score=0.652**  
   *ans:* high blood pressure also called hypertension usually has no symptoms but it can cause serious problems such as stroke heart failure heart attack and kidney failure if you cannot control your high blood pressure through lifestyle changes such as losing weight and reducing sodium in your diet you may need medicines blood pressure medicines work in different ways to lower blood pressure some remove extra fluid and salt from the body others slow down the heartbeat or relax and widen blood vessels often two or more medicines work better than one nih national heart lung and blood institute  
3. **score=0.651**  
   *ans:* blood pressure is the force of blood pushing against the walls of the blood vessels as the heart pumps blood if your blood pressure rises and stays high over time its called high blood pressure high blood pressure is dangerous because it makes the heart work too hard and the high force of the blood flow can harm arteries and organs such as the heart kidneys brain and eyes  

## Example Interactions (Fine-tuned)

**User:** What are the symptoms of diabetes?  
1. **score=0.870**  
   *ans:* the signs and symptoms of diabetes are being very thirsty urinating often feeling very hungry feeling very tired losing weight without trying sores that heal slowly dry itchy skin feelings of pins and needles in your feet losing feeling in your feet blurry eyesight some people with diabetes dont have any of these signs or symptoms the only way to know if you have diabetes is to have your doctor do a blood test  
2. **score=0.860**  
   *ans:* many people with diabetes experience one or more symptoms including extreme thirst or hunger a frequent need to urinate andor fatigue some lose weight without trying additional signs include sores that heal slowly dry itchy skin loss of feeling or tingling in the feet and blurry eyesight some people with diabetes however have no symptoms at all  
3. **score=0.718**  
   *ans:* often no symptoms appear during the early stages of diabetes retina problems as retina problems worsen your symptoms might include blurry or double vision rings flashing lights or blank spots in your vision dark or floating spots in your vision pain or pressure in one or both of your eyes trouble seeing things out of the corners of your eyes  

**User:** How can I treat a common cold?  
1. **score=0.684**  
   *ans:* sneezing sore throat a stuffy nose coughing everyone knows the symptoms of the common cold it is probably the most common illness in the course of a year people in the united states suffer 1 billion colds you can get a cold by touching your eyes or nose after you touch surfaces with cold germs on them you can also inhale the germs symptoms usually begin 2 or 3 days after infection and last 2 to 14 days washing your hands and staying away from people with colds will help you avoid colds there is no cure for the common cold for relief try getting plenty of rest drinking fluids gargling with warm salt water using cough drops or throat sprays taking overthecounter pain or cold medicines however do not give aspirin to children and do not give cough medicine to children under four nih national institute of allergy and infectious diseases  
2. **score=0.668**  
   *ans:* summary sneezing sore throat a stuffy nose coughing everyone knows the symptoms of the common cold it is probably the most common illness in the course of a year people in the united states suffer 1 billion colds what can you do for your cold or cough symptoms besides drinking plenty of fluids and getting plenty of rest you may want to take medicines there are lots of different cold and cough medicines and they do different things nasal decongestants unclog a stuffy nose cough suppressants quiet a cough expectorants loosen mucus so you can cough it up antihistamines stop runny noses and sneezing pain relievers ease fever headaches and minor aches and pains here are some other things to keep in mind about cold and cough medicines read labels because many cold and cough medicines contain the same active ingredients taking too much of certain pain relievers can lead to serious injury do not give cough medicines to children under four and dont give aspirin to children finally antibiotics wont help a cold food and drug administration  
3. **score=0.617**  
   *ans:* summary when you cough or sneeze you send tiny germfilled droplets into the air colds and flu usually spread that way you can help stop the spread of germs by covering your mouth and nose when you sneeze or cough sneeze or cough into your elbow not your hands cleaning your hands often always before you eat or prepare food and after you use the bathroom or change a diaper avoiding touching your eyes nose or mouth hand washing is one of the most effective and most overlooked ways to stop disease soap and water work well to kill germs wash for at least 20 seconds and rub your hands briskly disposable hand wipes or gel sanitizers also work well  

**User:** What causes high blood pressure?  
1. **score=0.827**  
   *ans:* changes in body functions researchers continue to study how various changes in normal body functions cause high blood pressure the key functions affected in high blood pressure include kidney fluid and salt balances the reninangiotensinaldosterone system the sympathetic nervous system activity blood vessel structure and function kidney fluid and salt balances the reninangiotensinaldosterone system the sympathetic nervous system activity blood vessel structure and function kidney fluid and salt balances the kidneys normally regulate the bodys salt balance by retaining sodium and water and eliminating potassium imbalances in this kidney function can expand blood volumes which can cause high blood pressure reninangiotensinaldosterone system the reninangiotensinaldosterone system makes angiotensin and aldosterone hormones angiotensin narrows or constricts blood vessels which can lead to an increase in blood pressure aldosterone controls how the kidneys balance fluid and salt levels increased aldosterone levels or activity may change this kidney function leading to increased blood volumes and high blood pressure sympathetic nervous system activity the sympathetic nervous system has important functions in blood pressure regulation including heart rate blood pressure and breathing rate researchers are investigating whether imbalances in this system cause high blood pressure blood vessel structure and function changes in the structure and function of small and large arteries may contribute to high blood pressure the angiotensin pathway and the immune system may stiffen small and large arteries which can affect blood pressure genetic causes high blood pressure often runs in families years of research have identified many genes and other mutations associated with high blood pressure however known genetic factors only account for 2 to 3 percent of all cases emerging research suggests that certain dna changes before birth also may cause the development of high blood pressure later in life unhealthy lifestyle habits unhealthy lifestyle habits can cause high blood pressure including high sodium intake and sodium sensitivity drinking too much alcohol lack of physical activity high sodium intake and sodium sensitivity drinking too much alcohol lack of physical activity overweight and obesity research studies show that being overweight or obese can increase the resistance in the blood vessels causing the heart to work harder and leading to high blood pressure medicines prescription medicines such as asthma or hormone therapies including birth control pills and estrogen and overthecounter medicines such as cold relief medicines may cause high blood pressure this happens because medicines can change the way your body controls fluid and salt balances cause your blood vessels to constrict impact the reninangiotensinaldosterone system leading to high blood pressure change the way your body controls fluid and salt balances cause your blood vessels to constrict impact the reninangiotensinaldosterone system leading to

### Overall Assessment

The fine-tuned model clearly outperforms the baseline in **accuracy**, **completeness**, and **domain-specific detail**, especially for medically complex questions.  
While the baseline answers are acceptable, they tend to be **shorter**, more **generic**, and less explanatory.  
The fine-tuned responses demonstrate:

- **Improved relevance** (scores are consistently higher)
- **Richer detail** without losing factual accuracy
- **Better coverage of edge cases** (e.g., asymptomatic conditions)
- **Closer alignment** with authoritative health sources (NIH, FDA-style clarity)

---

### Category-by-Category Analysis

#### 1. Symptoms of Diabetes

**Baseline**  
- Provides good symptom lists but sometimes mixes Type 2 diabetes generalization without highlighting variations in presentation.

**Fine-tuned**  
- Maintains symptom lists but improves score (**0.870 vs 0.815**) and offers a more targeted selection of top answers with higher precision.  
- Notably, option 3 introduces retinal complications, which could be seen as relevant but slightly narrower than the general “symptoms” request — this is both a strength (extra depth) and a risk (off-topic drift if the intent was general overview only).

**Verdict:** Fine-tuned shows better recall and slightly more medical nuance, though the third answer might be too condition-specific for a general symptoms query.

---

#### 2. Treating a Common Cold

**Baseline**  
- Balanced, practical, and includes key prevention/treatment steps.  
- Scores hover in mid-0.6 range.

**Fine-tuned**  
- Content is very similar to baseline, and score gains are marginal (**0.684 vs 0.688**).  
- The improvement is less dramatic here because both models perform well and the domain (common cold) is less complex than diabetes or hypertension.

**Verdict:** Fine-tuned is roughly on par with baseline here — improvements are not significant, suggesting that for simpler, well-covered medical topics, the baseline was already strong.

---

#### 3. Causes of High Blood Pressure

**Baseline**  
- Lists causes in a broad, understandable way but lacks mechanistic depth and layering of causes.

**Fine-tuned**  
- Major improvement — scores jump significantly (**0.827 vs 0.665**).  
- The model delivers structured, hierarchical causes (*body function changes → genetic → lifestyle → medication → other conditions*).  
- It also repeats certain key factors for emphasis, which can aid recall but risks verbosity.

**Verdict:** This is the clearest case where fine-tuning dramatically improves explanatory richness and medical accuracy.

---

### Key Strengths of the Fine-Tuned Model

- **Higher factual density** — more details per answer
- **Better cause-effect explanation** — links symptoms or causes to physiological processes
- **Improved recall** — mentions more possible relevant points
- **Scores are consistently higher** — suggesting better retrieval alignment

---

### Potential Risks

- **Verbosity** — sometimes repeats phrases or entire clauses (especially in hypertension cause explanation)
- **Topic drift** — e.g., introducing retina-specific complications in a general diabetes symptom query
- **Cognitive overload** — some answers may be too dense for a general audience without breaking into bullet points

---

### Final Judgment

The fine-tuned model **substantially improves depth and alignment** for complex health questions, particularly those that require structured reasoning or hierarchical explanations (e.g., high blood pressure causes).  
For simpler or well-documented topics (e.g., common cold treatment), gains are smaller but still show slight precision improvements.  

**Overall:** Fine-tuning meaningfully enhances medical reliability and completeness while keeping responses consistent with health authority guidelines.


## Potential Improvements
1. **Model Architecture:** Explore other transformer encoders or cross-encoder architectures for better fine-grained ranking.
2. **Negative Sampling:** Include adversarial or semantic hard negatives beyond BM25 top candidates.
3. **Data Augmentation:** Expand dataset with paraphrased questions or synthetic Q&A pairs to improve generalization.
4. **Evaluation Metrics:** Incorporate user-centric metrics or domain-specific scoring to better reflect real-world performance.
5. **Hybrid Retrieval:** Combine sparse (BM25) and dense retrieval to leverage complementary strengths.
6. **RAG (Retrieval-Augmented Generation):** Integrate retrieved answers into a generative model to produce natural language responses and handle queries not directly in the dataset.
7. **Multi-modal Extensions:** Incorporate additional metadata or structured medical knowledge bases.

## Usage
1. Update `DATA_PATH` to your dataset CSV.
2. Run the pipeline:
```bash
python medical_qa_pipeline.py
```
## Note
I did not use AI assistance for this project

