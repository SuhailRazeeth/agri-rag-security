[Date]

The Editors-in-Chief
Computers and Electronics in Agriculture
Elsevier

Dear Editors,

We are pleased to submit our manuscript entitled **"Secure and Faithful
Retrieval-Augmented Advisory Systems: A Domain-Specific Benchmark of Guardrail
Strategies against Direct and Indirect Prompt Injection in Agricultural
Economics"** for consideration as an original research article in *Computers and
Electronics in Agriculture*.

Large language models with retrieval-augmented generation (RAG) are being rapidly
adopted as low-cost digital advisory services for smallholder farmers, and this
journal has published a growing body of work on such systems. Almost all of that
work evaluates *capability* — accuracy, language coverage, cost. Our manuscript
addresses the complementary and, we argue, safety-critical question that has been
overlooked in the agricultural domain: **can such an advisory assistant be
manipulated into giving harmful advice?** In agriculture the stakes are concrete —
a fabricated subsidy figure, a manipulated market price, or an unsafe agrochemical
dose translates directly into financial or physical harm for users with limited
means to verify the answer.

To answer this, we built the first domain-specific security-and-faithfulness
benchmark for an agricultural-economics RAG advisory assistant and used it to
compare six guardrail strategies across three instruction-tuned models. The main
contributions and findings are:

1. A domain-grounded attack taxonomy and benchmark (direct and indirect
   prompt-injection attacks organised by agri-economic harm categories such as
   fabricated subsidy, price manipulation, unsafe dosage, and commercial
   steering).

2. A controlled, statistically tested comparison of six defenses, including
   Meta's published Llama Prompt Guard 2 and spotlighting. Our central result is
   that **input-side defenses — even a state-of-the-art injection classifier — do
   not significantly reduce indirect prompt injection**, because they inspect the
   benign farmer query rather than the poisoned retrieved document; only
   generation- and output-side defenses succeed (p < 0.0001).

3. Evidence that **no evaluated defense is a complete solution**: under an
   adaptive attacker the strongest defense degrades fivefold, and one class of
   "commercial steering" attacks resists every defense.

4. A jointly reported utility analysis (RAGAS faithfulness, over-defense, latency,
   cost) and a three-way validation of our LLM judge (against RAGAS, an
   independent second judge, and human annotation).

We believe these results are directly relevant to the readership of *Computers
and Electronics in Agriculture*: they provide practitioners building agricultural
advisory chatbots with concrete, evidence-based guidance on which guardrails to
deploy and which to distrust, and they establish a reproducible benchmark for the
community. To support reproducibility, the complete benchmark, attack taxonomy,
defense implementations, and raw results are publicly released on GitHub and
permanently archived on Zenodo (https://doi.org/10.5281/zenodo.22753856).

We confirm that this manuscript is original, has not been published elsewhere, and
is not under consideration by any other journal. All authors have read and
approved the submission and declare no conflict of interest. The study involved no
human or animal subjects; the small human-annotation component was performed by
the authors on model outputs and required no ethical approval.

Thank you for considering our work. We look forward to your response.

Sincerely,

[Corresponding Author Name], on behalf of all authors
Department of [Department], [Faculty], South Eastern University of Sri Lanka,
Oluvil, Sri Lanka
E-mail: [corresponding.author@seu.ac.lk]
