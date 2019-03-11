---
title: 'Cyphercat: A Python Package for Reproduceably Evaluating Robustness Against Privacy Attacks'
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
research efforts have increasingly focused on understanding security
vulnerabilities throughout the machine learning pipeline. 
Attack capabilities at inference time elucidate how the model output 
can be manipulated by having access to the training pipeline and
poisoning the training data, or by accessing the model and
manipulating images to fool the model [@goodfellow][@carlini]. 
Other attacks target machine learning as a service platforms, 
extracting the defining parameters of a target model [@tramer]. 
Less work has focused on privacy attacks, were nefarious agents 
can infer details of the training data from a targeted model [@mlleaks]. 
This has significant implications for user privacy and model sharing.
Fundamentally assessing model vulnerabilities to privacy attacks
remains an open-ended challenge, as current attack and defense
tactics are studied on a case by case basis.

Cyphercat is an extensible Python package for benchmarking privacy
attack and defense efficacy in a reproducible environment.
The Cyphercat application programming interface (API) allows users to test the 
robustness of a specified target model against several well-documented privacy
attacks [@mlleaks][@fredrikson2015model], which aim to extract details of the training data from the model.
Also included is the option to further assess the efficacy of several implemented defense methods.
The API is built on the PyTorch [@pytorch] machine learning library and 
provides access to well known image, audio, and text benchmark datasets used for machine learning applications.
The Cyphercat API includes the option to train on commonly used architectures, 
with subsequent assessment of attack and defense performance.
The package also enables users to introduce custom datasets and model architectures.

To use the API, a user must define a dataset, including data transformations,
and the desired architectures for the target model (the model being assessed for vulnerabilities)
and the attack model (the model used for generating an attack on the target model).
These are then fed into specified functions to initiate training, attacking and defending.
The source code for Cyphercat is available [here](https://github.com/Lab41/cyphercat/).


# References
