# Parameter Estimate 

- 확률 기반으로 데이터를 모델링 ( Parameter 를 추정할 때 사용하는 기법 )
- Target Population ( 모분포 ) -> sample ( 샘플링 된 데이터에서 모분포의 특징을 추정하는 것 )

## Point estimate 

- a single value of a statics

## interval estimate 

$$a < x < b$$ is an interval estimate of the population mean $$\mu$$

## Likelihood 

- 100 명 중 65명이 `successes` 인 binomial probability ( 이항 분포 )를 얻는 p 값을 찾는 것이 목표 
- this takes the data as fixed ( 데이터가 고정된 상태 ) and computes the probability of the data for a given $p$
- data = 65

- Likelihood 

$$
P\left(data | p\right)=\left(\begin{matrix}100 \\65\end{matrix}\right)
p^{65}(1-p)^{35}
$$  

## Maximum Likelihood Estimation ( MLE )

- Likelihood 가 최대가 되는 p 값을 찾는 것이 목표
- 이항분포의 그래프의 미분 0지점이 꼭지점 최대 부분이기 때문 
  
$$
\frac{d}{dp}P(\,data\,|\,p\,) = 0 \;for\, P
$$

- 예제 계산 
$$
\frac{d}{dp}\left(\begin{matrix}100 \\65\end{matrix}\right)
p^{65}(1-p)^{35}\;\\\\
65(1\,-\,p)\;=\;35p\\
65-65p=35p\\
p=\frac{65}{100}
$$

### Log Likelihood 

the log function turns multiplication into addtion, it's convinient to use the log of the likelihood function.

$$
\ln P\left(data | p\right)= \ln\,(\left(\begin{matrix}100 \\65\end{matrix}\right)
p^{65}(1-p)^{35})
$$  

- 위의 식이 log의 성질로 인해 곱이 덧셈으로 변환이 된다.

$$
\ln P\left(data | p\right)= \ln\left(\begin{matrix}100 \\65\end{matrix}\right)
+ \ln\,(p^{65}) + \ln\,(1-p)^{35}
$$  

- 로그의 미분은 다음과 같다

$$
\frac{d}{dx}(\ln\,x)\,=\,\frac{1}{x}
$$

- sample 크기에 따라서 추정된 파라메터를 신뢰할 수 있는지가 달라진다. 

