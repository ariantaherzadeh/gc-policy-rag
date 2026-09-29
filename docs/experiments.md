# Experiment log

Every run against the real Cohere API, in order: what was run, the raw output, and the
facts it shows. Newest at the bottom. Interpretation goes under **My notes** (owner-written).

Corpus at the time of these runs: 1 document (Directive on Automated Decision-Making, EN),
default `block` chunker, 83 chunks. Models: `embed-v4.0` (1536 dims), `rerank-v4.0-pro`,
`command-a-03-2025`.

| # | Date | What | API calls | Running total |
|---|---|---|---|---|
| 1 | 2026-09-27 | Embed the directive (ingest) | 1 | 1 |
| 2 | 2026-09-27 | Vector search, English question | 1 | 2 |
| 3 | 2026-09-27 | Vector search, French question (cross-lingual) | 1 | 3 |
| 4 | 2026-09-27 | Where did clause 6.1.1 rank? (cached question) | 0 | 3 |
| 5 | 2026-09-29 | First full answer: failed, chunk ids contained spaces | 2* | 5 |
| 6 | 2026-09-29 | Full answer **with** rerank | 2 | 7 |
| 7 | 2026-09-29 | Full answer **without** rerank | 1 | 8 |

\* Rerank succeeded, then Chat rejected the request. Counted conservatively; the rejected call may not count.

---

## 1. Embed the directive

```
$ uv run python -m app.ingest
chunker: block, 83 chunks from 1 document(s)
embed:   embed-v4.0 (1536 dims), 0 cached, 83 new -> 1 API call(s)
embedded: 1 API call(s) made

doc_id            chunks  embedded
dadm-en               83        83

$ uv run python -m app.ingest --estimate      # afterwards
embed:   embed-v4.0 (1536 dims), 83 cached, 0 new -> 0 API call(s)
```

Facts: 83 chunks fit in one batch (max 96). A second run costs 0 calls (content-hash cache).

**My notes:**

---

## 2. Vector search, English question

Question: *What must a department do before an automated decision system goes into production?*

```
$ uv run python -m app.retrieve "What must a department do before an automated decision system goes into production?"
 1. 0.644  dadm-en:6.3.14:0           Obtaining the appropriate level of approvals prior to the production o
 2. 0.581  dadm-en:Appendix A:4       production: An automated decision system is in production when it is i
 3. 0.581  dadm-en:6.3.1:0            Before an automated decision system is in production, testing the data
 4. 0.548  dadm-en:6.2.6.2:0          As part of this access, the department responsible for the automated d
 5. 0.539  dadm-en:6.2.6.1:0          The department responsible for the automated decision system retains t
 6. 0.538  dadm-en:6.3.12:0           Consulting with the department’s legal services from the concept stage
 7. 0.538  dadm-en:4.2.3:0            Data and information on the use of automated decision systems in depar
 8. 0.513  dadm-en:5.1:0              This directive applies to any automated decision system in production
 9. 0.510  dadm-en:6.3.13:0           Ensuring that the automated decision system allows for human involveme
10. 0.501  dadm-en:6.3.7:0            Consulting the appropriate qualified experts to review the automated d
```

(Chunk ids were later changed from `Appendix A` to `Appendix-A`; see run 5.)

Facts:
- #2 is the *definition* of "production" (Appendix A), which shares words with the question but states no requirement.
- #4 and #5 are about proprietary software licences.
- Clause 6.1.1 (publish an algorithmic impact assessment before production) is not in the top 10.
- Top-10 scores span 0.501–0.644.

**My notes:**

---

## 3. Vector search, French question (cross-lingual)

Question: *Un examen par les pairs est-il obligatoire pour un système de niveau II?*
(Is peer review mandatory for a Level II system?) The corpus is English only.

```
$ uv run python -m app.retrieve "Un examen par les pairs est-il obligatoire pour un système de niveau II?"
 1. 0.416  dadm-en:Appendix C:6       Requirement: Approval for the system to operate (section 6.3.14) Level
 2. 0.416  dadm-en:Appendix C:2       Requirement: Peer review (section 6.3.7) Level I: None Level II: Consu
 3. 0.348  dadm-en:8.3.5:0            The heads of agents of Parliaments are responsible for approval of lev
 4. 0.346  dadm-en:Appendix C:5       Requirement: Ensuring human involvement (section 6.3.13) Level I: The
 5. 0.314  dadm-en:Appendix B:1       Level: II Description: The context in which the system is operating li
 6. 0.308  dadm-en:6.2.6.2:0          As part of this access, the department responsible for the automated d
 7. 0.303  dadm-en:Appendix C:4       Requirement: Training (section 6.3.9) Level I: Role-based training on
 8. 0.296  dadm-en:Appendix C:1       Requirement: Explanation (section 6.2.3) Level I: In addition to any a
 9. 0.280  dadm-en:Appendix C:3       Requirement: Gender-based Analysis Plus (section 6.3.8) Level I: None
10. 0.279  dadm-en:Appendix C:0       Requirement: Notice (sections 6.2.1–6.2.2) Level I: Plain language not
```

Facts:
- A French question retrieved the English Appendix C "Peer review" row at #2 (tied with #1). That row contains the answer (Level II: consult at least one qualified expert).
- Scores span 0.279–0.416, lower than the English question's 0.501–0.644, even though the top French match is relevant and the English #2 is not.

**My notes:**

---

## 4. Where did clause 6.1.1 rank?

Same English question as run 2, top 50 (cached, 0 calls).

```
$ uv run python -m app.retrieve "What must a department do before an automated decision system goes into production?" --top-k 50 | grep -E "6\.1\.1|6\.3\.7|Appendix C:2"
10. 0.501  dadm-en:6.3.7:0            Consulting the appropriate qualified experts to review the automated d
34. 0.406  dadm-en:6.1.1:0            Completing, approving and publishing the final results of an algorithm
44. 0.332  dadm-en:Appendix C:2       Requirement: Peer review (section 6.3.7) Level I: None Level II: Consu
```

Facts: 6.1.1 is at rank **34 of 50**: inside what rerank sees, outside the top 8 that Command sees without rerank.
Its text never names who must act or the topic heading; both live in the lead-in paragraph and heading path, which the default chunker leaves out.

**My notes:**

---

## 5. First full answer: failed

```
$ uv run python -m app.chat "What must a department do before an automated decision system goes into production?" --show-context
cohere.errors.bad_request_error.BadRequestError: ... status_code: 400, body: {..., 'message': 'invalid request: document id contains whitespace for document index 6'}
```

Facts:
- Rerank accepted the passages; Chat rejected the request because a chunk id (`dadm-en:Appendix C:1`) contained a space.
- Offline tests passed: the fake clients didn't know this API rule.
- Fix: chunk ids use hyphens (`dadm-en:Appendix-C:1`), with a test. Re-ingest cost 0 calls (embeddings cached by text).

**My notes:**

---

## 6. Full answer with rerank

```
$ uv run python -m app.chat "What must a department do before an automated decision system goes into production?" --show-context
Before an automated decision system goes into production, a department must:
- Test the data, information, and underlying model for accuracy, unintended biases and all factors that may unintentionally or unfairly impact the outcomes or violate human rights and freedoms.[1]
- Consult the appropriate qualified experts to review the automated decision system, algorithmic impact assessment and supporting documentation, and publish the complete review or a plain language summary.[2]
- Complete, approve and publish the final results of an algorithmic impact assessment in an accessible format on the Open Government Portal.[3]
- Obtain the appropriate level of approvals.[4]
- Consult with the department’s legal services from the concept stage of an automation project to ensure that the use of the automated decision system is compliant with applicable legal requirements.[5]
- Validate that the data used to train the automated decision system and the data input into the system are relevant, accurate, up-to-date, and in accordance with the Policy on Service and Digital and the Privacy Act.[6]
- Conduct risk assessments during the development and maintenance of the automated decision system and implement appropriate information management and information technology security protections, in accordance with the Policy on Government Security and the Policy on Service and Digital.[7]

[1] Directive on Automated Decision-Making, section 6.3.1  (dadm-en:6.3.1:0)
[2] Directive on Automated Decision-Making, section 6.3.7  (dadm-en:6.3.7:0)
[3] Directive on Automated Decision-Making, section 6.1.1  (dadm-en:6.1.1:0)
[4] Directive on Automated Decision-Making, section 6.3.14  (dadm-en:6.3.14:0)
[5] Directive on Automated Decision-Making, section 6.3.12  (dadm-en:6.3.12:0)
[6] Directive on Automated Decision-Making, section 6.3.5  (dadm-en:6.3.5:0)
[7] Directive on Automated Decision-Making, section 6.3.10  (dadm-en:6.3.10:0)

Passages sent to Command (rerank score):
  0.961  dadm-en:6.3.1:0            Before an automated decision system is in production, testin
  0.955  dadm-en:6.3.7:0            Consulting the appropriate qualified experts to review the a
  0.937  dadm-en:6.1.1:0            Completing, approving and publishing the final results of an
  0.934  dadm-en:6.3.14:0           Obtaining the appropriate level of approvals prior to the pr
  0.914  dadm-en:6.3.12:0           Consulting with the department’s legal services from the con
  0.864  dadm-en:6.3.5:0            Validating that the data used to train the automated decisio
  0.859  dadm-en:Appendix-C:1       Requirement: Explanation (section 6.2.3) Level I: In additio
  0.858  dadm-en:6.3.10:0           Conducting risk assessments during the development and maint

API calls: embed 0, rerank 1, chat 1 = 2
```

**My notes:**

---

## 7. Full answer without rerank

```
$ uv run python -m app.chat "What must a department do before an automated decision system goes into production?" --no-rerank --show-context
Before an automated decision system goes into production, a department must:
- Test the data, information, and underlying model for accuracy, unintended biases and all factors that may unintentionally or unfairly impact the outcomes or violate human rights and freedoms.[1]
- Obtain the appropriate level of approvals, as prescribed in Appendix C.[2]
- Consult with the department’s legal services from the concept stage of an automation project to ensure that the use of the automated decision system is compliant with applicable legal requirements.[3]

[1] Directive on Automated Decision-Making, section 6.3.1  (dadm-en:6.3.1:0)
[2] Directive on Automated Decision-Making, section 6.3.14  (dadm-en:6.3.14:0)
[3] Directive on Automated Decision-Making, section 6.3.12  (dadm-en:6.3.12:0)

Passages sent to Command (vector score):
  0.644  dadm-en:6.3.14:0           Obtaining the appropriate level of approvals prior to the pr
  0.581  dadm-en:Appendix-A:4       production: An automated decision system is in production wh
  0.581  dadm-en:6.3.1:0            Before an automated decision system is in production, testin
  0.548  dadm-en:6.2.6.2:0          As part of this access, the department responsible for the a
  0.539  dadm-en:6.2.6.1:0          The department responsible for the automated decision system
  0.538  dadm-en:6.3.12:0           Consulting with the department’s legal services from the con
  0.538  dadm-en:4.2.3:0            Data and information on the use of automated decision system
  0.513  dadm-en:5.1:0              This directive applies to any automated decision system in p

API calls: embed 0, rerank 0, chat 1 = 1
```

### Runs 6 vs 7 side by side

| | With rerank (run 6) | Without rerank (run 7) |
|---|---|---|
| Requirements in the answer | 7 | 3 |
| 6.1.1 (algorithmic impact assessment) in context | Yes, rank 3 | No (rank 34 in vector search) |
| 6.3.7 (peer review) in context | Yes, rank 2 | No (rank 10 in vector search) |
| Passages in context that state no pre-production requirement | 1 (Appendix-C:1, Explanation) | 5 (Appendix-A:4, 6.2.6.2, 6.2.6.1, 4.2.3, 5.1) |
| Score range of the 8 passages | 0.858–0.961 (rerank relevance) | 0.513–0.644 (cosine similarity) |
| Every claim cited | Yes | Yes |
| Claims not supported by the context | None observed | None observed |

Single question: an illustration, not a measurement. The eval harness measures this across many questions.

**My notes:**
