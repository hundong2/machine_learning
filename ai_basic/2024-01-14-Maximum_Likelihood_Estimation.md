---
title: "Maximum Likelihood Estimation"
permalink: /posts/maximum_likelihood/
excerpt: "Maximum Likelihood Estimation."
redirect_from:
  - /theme-setup/
toc: true
---

# Maximum Likelihood Estimation

- 모분포 위에서 파라메터를 추정할 때 추정하는 기법 
- 이때 N개의 데이터들은 각각 독립이기때문에 다음의 식이 나올 수 있다.
- Bernoulli(p) distribution을 따른다는 가정.  P, 1-P 으로 부터 sampling 되었을 때 

Finding $$\theta$$ that maximized $$P(X1, ..., Xn; \theta)$$  
We have $$P(Xi;\theta)$$

$$
P(X1,...,Xn;\theta)\,=\,\prod_{i} P(Xi;\theta)\\
\downarrow\\
\log{}{P(X1,...,Xn;\theta)}\,=\,\sum_{i}\log{P(Xi;\theta)}
$$

- 이렇게 로그 기반의 Likelihood 식에서 $\theta$에 대해 derivative하여 0이 되는 값을 찾는 경우가 최대인 Probability를 찾는 것이 목표


- Solving $$\rightarrow\frac{\theta}{\partial\theta}log\sum_{i}{P(Xi;\theta)=0}$$

- Log-likelihood is a monotonic function of the like lihood 
- likelihood에 Log를 씌우더라도 그래프에서의 maximum은 변하지 않는다.  
- Log를 씌웠을 때 효율이 좋다.

$$
\hat{p}=argmax_{p}\sum_{i}X_{i}\log{p}\,+\,\sum_{i}log{(1-p)}\\
\downarrow\\\,solving\\
\downarrow
$$
$$
\frac{\theta}{\partial\theta}log\sum_{i}{P(Xi;\theta)=0}
\\
$$
$$
\frac{\sum_{i}X_i}{\hat{p}}-\frac{n\,-\,\sum_{i}X_i}{1\,-\,\hat{p}}\,=\,0\;\Rightarrow\;\hat{p}\,=\,\frac{\sum_{i}(X_i)}{n}\;(n\,=\,\sum_{i}(1))
$$

## Gaussian Random variable

$$
X \sim \mathcal{N}(\mu,\,\sigma^{2})\,.
$$

[gaussian](https://en.wikipedia.org/wiki/Normal_distribution#/media/File:Normal_Distribution_PDF.svg)  

- MLE of $$\mu$$
$$
f_{x};(x;\,\mu,\,\sigma)\,=\,\prod_{i}f_{xi}(x_{i}; \mu, \sigma)\\
\downarrow \\
$$
$$
\ln\,f_{x}(x;\mu,\sigma)=\sum_{i}\ln\,f_{xi}(x_{i};\mu,\sigma)
$$

$$
f_{X}(X;\mu,\sigma)=\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(x_{i}-\mu)^{2}}{2\sigma^{2}}}
$$

- What is the MLE of $\mu$

$$
\ln\,f_{X}(X;\mu,\sigma)=\sum_{i}\ln\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(x_{i}-\mu)^{2}}{2\sigma^{2}}}\\
\Downarrow
$$

$$
\ln\,f_{X}(X;\mu,\sigma)=\sum_{i}\ln\frac{1}{\sqrt{2\pi\sigma^{2}}}+\sum_{i}\ln\,e^{-\frac{(x_{i}-\mu)^{2}}{2\sigma^{2}}}\\
\Downarrow
$$

$$
\ln\,f_{X}(X;\mu,\sigma)=-n\ln\,\sqrt{2\pi\sigma^{2}}+\sum_{i}\ln\,e^{-\frac{(x_{i}-\mu)^{2}}{2\sigma^{2}}}\,(n=\sum_{i})\\
\Downarrow
$$

$$
\ln\,f_{X}(X;\mu,\sigma)=-n\ln\,\sqrt{2\pi\sigma^{2}}-\sum_{i}\frac{(x_{i}-\mu)^{2}}{2\sigma^{2}}\,(n=\sum_{i},\ln{e}=1)
$$

$$
\Rightarrow Solving for \frac{\partial}{\partial\mu}\ln\,f_{X}(X;\mu,\sigma)=0
$$

$$
\frac{\partial}{\partial\mu}\sum_{i}\frac{(x_{i}-\mu)^{2}}{2\sigma^{2}}=0
$$
- 합성 함수에 대한 미분을 진행
- n은 개수가 된다. 
$$
\sum_{i}x_{i}-n\hat{\mu}=0\;\Rightarrow\;\;\hat{\mu}=\frac{\sum_{i}x_{i}}{n}
$$