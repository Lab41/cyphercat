---
title: 'CypherCat: A Python Package for Reproduceably Evaluating Adversarial Robustness
tags:
  - Python
  - machine learning
  - adversarial attacks
  - model inversion
authors:
  - name: Zigfried Hampel-Arias
    orcid: 0000-0003-0253-9117
    affiliation: "1"
    email: zhampel.github@gmail.com
  - name: Nina Lopatina
    orcid: 0000-0001-6844-4941
    affiliation: "1"
    email: ninalopatina@gmail.com
  - name: Michael Lomnitz
    orcid: 0000-0001-5659-3501
    affiliation: "1"
    email: mllomnitz@gmail.com
  - name: Felipe A Mejia
    orcid: 
    affiliation: "1"
    email: felipe.a.mejia@gmail.com
affiliations:
 - name: Lab41 -- an InQTel Lab, Menlo Park, CA, USA
   index: 1
date: DD Mon 2019
bibliography: paper.bib
---

# Summary

While evaluating a machine learning model's performance is well-established, assessing potential vulnerabilities of models to attack remains open-ended.
Furthermore, addressing methods to curtail attacks effectively requires additional investigation.
We present an application programming interface (API) called CypherCat,
a standalone, extensible package with the aim of providing a toolkit to benchmark attack and defense efficacy in a reproduceable manner.
The API allows users to test the robustness of models against a variety of well-documented attacks, including the option to implement various defenses.
CypherCat provides users the flexibility to provide a dataset and a pretrained model, as well as access to standard datasets and the option to train
desired architectures via the API, with subsequent assessment of attack and defense performance.
We comment on the capabilities for image and audio data types, predefined methods and metrics of attacks and defenses,
and the implementation of CypherCat specific to the Python ecosystem, outlining its usability and flexibility.
[@mlleaks]

# Acknowledgements

The authors acknowledge support from 
We also acknowledge the financial support provided by the 

# References
