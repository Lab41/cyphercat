---
title: 'Cyphercat: A Python Package for Reproduceably Evaluating Adversarial Robustness
tags:
  - Python
  - machine learning
  - adversarial attacks
  - model inversion
  - model privacy
  - model robustness
authors:
  - name: Maria A. Barrios
    orcid:
    affiliation: "1"
    email: mbarrios@iqt.org
  - name: Paul Gamble
    orcid:
    affiliation: "1"
    email: pgamble@iqt.org
  - name: Zigfried Hampel-Arias
    orcid: 0000-0003-0253-9117
    affiliation: "1"
    email: zhampel.github@gmail.com
  - name: Nina Lopatina
    orcid: 0000-0001-6844-4941
    affiliation: "1"
    email: Nlopatina@iqt.org
  - name: Michael Lomnitz
    orcid: 0000-0001-5659-3501
    affiliation: "1"
    email: mllomnitz@gmail.com
  - name: Felipe A. Mejia
    orcid: 0000-0001-6393-8408
    affiliation: "1"
    email: felipe.a.mejia@gmail.com
  - name: Lucas Tindall
    orcid: 0000-0003-1395-4818
    affiliation: "1"
    email: ltindall@ucsd.edu
affiliations:
 - name: Lab41 -- an InQTel Lab, Menlo Park, CA, USA
   index: 1
date: DD MM 2019
bibliography: paper.bib
---

# Summary

With the proliferation of machine learning in everyday applications,
research efforts have increasingly focused on understanding the vulnerabilities of
machine learning models to privacy attacks.
For example, this can involve extracting information regarding the defining parameters of a _target_ model
or inferring details of data samples used to train the model.
These types of attacks pave the way for nefarious agents to infer potentially private information
from the training data or to manipulate the intended use of a trained model, 
for example by forcing the model to produce a desired output.
Fundamentally assessing model vulnerabilities to privacy attacks remains an open-ended challenge,
as current attack and defense tactics are studied on a case by case basis.


``Cyphercat`` is an extensible Python package for benchmarking privacy attack and defense efficacy
in a reproduceable manner.
The ``Cyphercat`` application programming interface (API) allows users to test the robustness a specified 
target model against several well-documented privacy attacks (such as those presented in [@mlleaks], [@fredrikson2015model])
that extract details of the training data from the model, with the option to assess defenses.
The API is based on the PyTorch [@pytorch] machine learning library, provides access to datasets 
traditionally used for benchmarking machine learning models, and the option to train commonly used 
architectures via the API, with subsequent assessment of attack and defense performance.
The package also permits users to introduce custom datasets in the image, audio, and text data type domains,
as well as custom architectures for target, attack, and defense models.


``Cyphercat`` is a flexible framework designed for machine learning practitioners to test model vulnerabilities
via various methods of attack and defense covering several data types.
Details regarding the ``Cyphercat`` API, its implementation relative to the Python ecosystem, 
including further information on implemented datasets, attack and defense methods, 
and performance metrics are found in the online documentation.


# Acknowledgements

The authors acknowledge support from 
We also acknowledge the financial support provided by the 

# References
