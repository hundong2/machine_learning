---
title: "The Concept of Logistic Regression "
permalink: /posts/LogisticRegressionConcept/
excerpt: "The Concept of Logistic Regression."
redirect_from:
  - /theme-setup/
toc: true
---
# The Concept of Logistic Regression 

- Probabilistic View for a Linear Classifier 
- Output as $P(y=1\;|\;x)$

$$
f(x)=
\begin{cases}
1\;(Red) & \quad \text{if $f(x) \geq 0$}\\ 
0\;(Blue) & \quad \text{otherwise}
\end{cases} \quad \Rightarrow \quad
h(x) = \frac{1}{1+\exp{(-f(x))}} \quad 
$$

$$
1+\exp{(-f(x))} \Rightarrow Sigmoid \; or\; Logistic\;function \\
$$
- 장점 : 
  - miss classification에서 각 점들에 대해 서로 다른 가중치를 줄 수 있다. 
  - 미분이 불가능한 특정 구간을 미분 가능하도록 만들 수 있다.    

## Sigmoid 

[Sigmoid wiki](https://en.wikipedia.org/wiki/Sigmoid_function#/media/File:Error_Function.svg)  

- sigmoid function is a S-curve shape 
- Bounded ( y 값이 bound ) $\rightarrow ( -1 < y < 1 )$
- Differential ( 모든 점에 대해 미분이 가능 )
- Defined for all real input $\rightarrow (-\infin<x<\infin )$
- With positive derivative ( 주어진 미분값들이 0보다 크다. ) 

### Logistic function 

$
\sigma(x)=\frac{L}{1+e^{-k(x-x_{0})}}
$

- $x_{0}\quad:$ the midpoint of the x-value.
- $L\quad\;:$ the curve's maximum value. 
- $k\quad\;\;:$ the steepness of the curve.

- Logistic function은 위의 $\sigma(x)$의 $x$에 $f(x)$를 대입하여 문제를 해결
- 이때 $f(x)=w^Tx$