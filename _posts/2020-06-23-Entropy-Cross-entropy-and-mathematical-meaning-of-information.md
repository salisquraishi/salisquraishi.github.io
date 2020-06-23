---
title:  "Entropy Cross-entropy and mathematical meaning of information"
layout: post
mathjax: true
---

> I this article I tried to present a classical and in-depth explanation of *entropy* and *cross-entropy* and their relationships with uncertainty or probability.   


## Information and Uncertainty
Would you really be surprised if I told you that tomorrow sun will rise from east? In another scenario, how would you react if you came to know that a military chopper is going to land on your neighbourhood? There is no doubt that former event is certainly going to happen, and your reply might be, "Well !!, What's new into that?" and in a latter situation you might come out of your home exclaiming "Wait.. What.. Chopper!! When, Why, Does anyone have any clue?". Now think again, which event carries more information for you? Here, considering information as a meaning rather than as an amount of information, would really help to establish a connection between information and uncertainty. 

We can clearly be convinced from above examples that probability(a measure of uncertainty) has a close association with information (i.e. meaning or conclusion we derive from the information we receive). The very fact sun will rise from east tomorrow is a less surprised more probable (in fact a certain event unless today is a dooms day) event which gives no new information or no information to be precise. In another case, the pleasantly surprised really improbable event that a chopper is going to land in your locality carries a lot of new information such as why it's landing, where and how.

Once we settle down with the fact that there is no information if there is no uncertainty or more uncertainty brings more information. We can say that information is inversely proportional to probability, i.e. a less probable event has more information. 

$$Information \propto \frac{1}{Probability}$$

Cloude Shannon is the father of mathematical disciple known as *Information Theory* which mainly deals with all the aspects of communication and storage of information. In 1948, Shannon presented a breakthrough and celebrated paper on information theory titled *"A Mathematical Theory of Communication"*. Shannon's quest was to quantify the amount of information from uncertainty. In other words, his effort was to find a measure of information and define that measure of information as the measure of uncertainty and he zeroed on to a really convincing solution - measure information as a function of probability.

$$H=-\sum_i p_ilog(p_i)$$

where $p_i$ is probability of an event, message, status, outcomes whatever. 

Shannon needed a new unit of measure and a name. He coined the term "*bits*" which stands for "*binary digits*", a smallest possible quantity of of information, For example, tossing a coin results in either head or tail(without bayesian thought process of course), to describe the outcome a minimum one bit is required which would be either 1 or 0. Also, he called H "*Entropy*", a unit of information which would be measured in "*bits*". If you want to give one liner statement about entropy, "*Entropy is information*" would be very precise choices of words.

Let's next try to understand Shannon's formula of information (Entropy) in the light of probability theory.

## Entropy and Information
As we know a random variable X describes the outcome of an experiment as a numerical value. For example flipping a coin, the outcome $\{H,T\}$ makes random variable $X$ takes $\{0,1\}$ respectively. The probability mass function (PMF) associated with any random variable $X$ is defined as $$p(x)$$. 

The entropy $H(X)$, which is a measure of uncertainty of a discrete random variable $X$ is defined by:

$$H(X) = -\sum_xp(x)log_2p(x)$$

*Note: Conventionally $H(p)$ is a more commonly used notation used in place of $H(X)$.*

Another way to describe $H(X)$ is that it is an expected value of a random variable $log(1/p(x))$. My advice would be to take a pause here and review the concept of *Expectations* in probability.

Now, let me tell you the intuition behind the reason of calculating entropy $H(X)$ of a random variable $X$. It quantifies the average description length needed to describe a random variable in bits, which is also a minimum bits required to describe a random variable (Bingo!! The first data compression idea, see example that follows in the next paragraph). 

Another important fact is that it gives you a whole new way comparing two or more probability distributions. If entropies of two probability distributions $p(x)$ and $q(y)$ are $H(p(x))$ and $H(q(y))$ respectively, and if former is greater or smaller than the latter then this difference in their entropies is also a quantity which tells us that how far or close both distributions are. Here comes the idea of *cross entropy* (more on this soon).  

Let's try to understand the average amount of information required to describe a random variable with an example. We know that $n$ binary digits can convey at max $2^n$ independent information (do the math please). Suppose you have a fitness tracker which tracks different health measurements of your body such as blood pressure, pulse rate, heart rate etc and based of those several measurements the device categorizes body's overall health status in $8$ different categories, and sends the health status to the cloud storage for further analysis.Also, keep in mind that eight different status is nothing but eight possible outcomes from a probability distribution of an event. 

Now, consider two assumptions and entropies associated with them below, 

Assumption A: All $8$ events are equaly likely.

$$p(x) = \{\frac18, \frac18, \frac18, \frac18, \frac18, \frac18, \frac18,\frac18\}$$

Therefore, the entropy in this case (in bits) is 3 bits. 

$$H(X) = -\sum_{x=1}^8p(x)log_2(p(x)) = -\sum_{x=1}^8(\frac18)log_2(\frac18) = 3 bits$$

Assumption B: All $8$ events are NOT equally likely. 

$$p(x) = \{\frac12, \frac14, \frac18, \frac1{16},
\frac1{64}, \frac1{64}, \frac1{64}, \frac1{64}\}$$

therefore, the entropy in this case is 2 bits. 

$$H(X) = -\sum_{x=1}^8p(x)log_2(p(x))$$

$$ 
= \frac12log\frac12 + \frac14log\frac14 + \frac18log\frac18 + \frac1{16}log\frac1{16}
+ \frac1{64}log\frac1{64} + \frac1{64}log\frac1{64} + \frac1{64}log\frac1{64}
$$  


$=2bits$

The important conclusion we can draw from our study of above two assumptions is that if we know the probability of events then there is an optimised way to describe the information, in above example we can see that there could be a way to encode (i.e. describe) the same information in lesser amount of bits. If we could find the way to do that then this will save us one bit per message per transmission. In above example, if tracker sends the status information to cloud every minute then we will only 1440\*2=2880 bits per day instead of 1440\*3=4320 bits and we will be able to save 4320 - 2880 = 1440 bits per day in network bandwidth and storage. 

So one simple way to achieve that is to describe less probable health status in a longer string of bits and more probable one in a shorter one. In that way frequent health status will be transmitted more often and will use less bandwidth in a long run and the same applies to less frequent health status. So the idea is simple, choose variable length strings of bits for different outcomes. 

The entropy for above example has given us the estimate that 2 bits per message transmission is the average information length we would consume in a long run. So lets encode the eight possible health status (outcomes) in such a way that the average length of bit strings is 2 bits. We can choose 1, 2, 3, 4 and 6 bits long string to represent those status in binary digits. Take a look at one sample of such encoded health status arrangement below.

$$A : 0$$<br>
$$B : 01$$<br>
$$C : 0001$$<br>
$$D : 0101$$<br>
$$E : 010101$$<br>
$$F : 101010$$<br>
$$G : 111000$$<br>
$$H : 111111$$<br>

since the average length of all eight encoded status would come as 2 (1+2+3+4+6/8 = 2), and going with above choice we can encode more frequent events (high probable outcome) in shorter strings, say 1 or 2 bits string and more frequent (less probable outcome) in a longer strings, say 4 or 6 bit strings. For instance, outcomes with probabilities 1/2 and 1/4 could be represented in one bit such as 0 and 1 respectively and outcomes with probabilities 1/16 and 1/64 in 6 bits such as 111000 101010 respectively.

## Cross-entropy and probability distributions
One thing should be clear by now is that entropy is a meausure of average uncertainty in the random variable or the amount of information required on average to describe the random variable. We can also say that entropy bridges between information and probability(i.e. uncertainty). 

If we want to focus our attention on two probability distributions rather than just one, then we may like to ask a question, how similar or dissimilar both distributions are? Having gained some knowledge about entropy, we can compare them with their respective entropies. For instance, $H(p)$ and $H(q)$ are the entropies of two sample probability distributions p and q which is assumed to have come from the same population. Therefore, there is is possibility that both entropies will vary, it depends how well the sampling has been done, but that's not our focus here, our focus is that if p represents true population probability better than q does then the entropy of q $H(q)$ would be higher than that of p, we just want to quantify that difference. Mathematically we can write it as below. 

$$H(q) = H(p) + Difference$$

That means if we were able to guage the difference(i.e. distance) we can compare two probabilities. If the difference is close the Zero then we can conclude that both the probabilities are almost same and represents true population distribution. 

*Relative Entropy* also knows as *Kullback-Leibler distance* measures the distance between two probability distribution and is defined as

$$D(p||q) = \sum_xp(x)log(p(x)/q(x))$$

That makes sense, as we can see if $p = q$, then $D(p\|\|q) = 0$

Another perception of relative entropy, could be that it is a *measure of the inefficiency* of interpreting q as a true representation of a population distribution when in fact it is p. The price we pay for that wrong assumption is that we encode the information (entropy in bits) based on q distribution which will end up in higher bits estimation.

$$H(q) = H(p) + D(p||q)$$

$$H(q) = -\sum p(x)log(p(x)) + \sum p(x)log(p(x)/q(x))$$

$$H(q) = -\sum p(x)log(p(x)) + \sum p(x)log(p(x)) - \sum p(x)log(q(x))$$

$$H(p,q) = -\sum p(x)log(q(x))$$

Above equation is called *Cross Entropy* or *Kullback-Leibler Divergence* and is a function of two probability distribution variables p and q, hence the change in notation from $H(q)$ to $H(p,q)$. This equates to $H(p)$ when $p=q$, which is nothing but the *entropy* of true representation of population distribution, which is p. The important point to note is that $H(p)$ is the lower bound of cross entropy. 

## Loss function in Machine Learning
instead of *Machine Learning* I wanted to use the term *Statistical Learning*  but sometime it's better to prefer popularity over literature.

Cross entropy found it's way into the world of Machine Learning. This is cleverly used in a classification algorithm to estimate the difference between two probability distribution and is commonly referred as *loss*. The training data which is assumed to be a true representation of the population distribution is compared with the less perfect probability distribution estimated by the model. A greater difference in probability distributions implies that model's ability to predict correct distribution (i.e. class) is poor therefore the model weights needs to be tuned with some optimisation algorithm such as *Gradient Descent* until that difference could me made as minimum as possible.

In any classification problem a model is presented with training data which is a combination of *features* and *labels*. Let's consider only binary classification problem where machine learning model's aim is to estimate the probability of a positive class for a given data point. $p$ represents a true population distribution and model's estimated probability distribution is defined by say $\hat{p}$ (read p-hat). What we want for our model is to give us $\hat{p}$ as close to $p$ as possible. i.e. we want $H(\hat{p})$ to reach as close to $H(\hat{p})$ as possible. 

Therefore, *Cross Entropy* is the way forward. We keep measuring the difference between $p$ and $\hat{p}$ in a regular interval, that is called training cycle or epoch. Let's go through a formal equation of cross entropy for a binary classification model. 

Below is our cross entropy equation for two probability distributions p and q. 

$$H(p,q) = -\sum_{x \in \{1,0\}} p(x)log(q(x))$$
<br>

It's Bernoulli distribution with probability of success $p$ and failure $1-p$. 

$$p(x)=p(X=x) = p^x(1-p)^{1-x}; x \in  \{0,1\}$$
<br>

Similarly for model's prediction it's $\hat{p}$ and $1-\hat{p}$. 

$$q(x) = q(X=x) = \hat{p}^x(1-\hat{p})^{1-x}; x \in  \{0,1\}$$
<br>

Let's put there probabilities together and expand $H(p,q)$. 

$$H(p,\hat{p}) = -\lbrack(p)log(\hat{p}) - (1-p)log(1-\hat{p})\rbrack$$

Above equation which is a function of two probability distributions needs to be minimized and commonly referred as *loss function* or *cross entropy loss* to be very precise with a new notation. 

$$L_{CE}(y,\hat{y}) = -\lbrack (y)log(\hat{y}) - (1-y)log(1-\hat{y})\rbrack$$

One immediate benefit of this new name is that it reminds us our main goal - that it's actually a loss which needs to be minimized. The function is at it's minimum when $y =$ $\hat{y}$ or in other words when $y$ $\rightarrow$ $\hat{y}$.

Let's test it through one example, suppose for a given data point $x$ the true label is 1 and our one model estimated the probability that it belongs to that label as 0.7 and another model estimated as 0.3. 

In the first case (when it's a good model),

$y=1$ and $\hat{y}=0.7$, therefore the the loss $L(y,\hat{y})=0.357$

In the second case (When it's a bad model),

$y=1$ and $\hat{y}=0.3$, therefore the the loss $L(y,\hat{y})=1.204$

It seems perfect, loss function showing greater loss when model made a poor judgement and that's all we need to assess the model, that means second model needs lot of tuning on weights.  

## Summary
Entropy is information or missing information, it's a measure of uncertainty and it has it's root on thermodynamics. It was mathematician John von Neumann who had suggested the name *Entropy* to Cloude Shannon when he was about the publish his paper on *Information Theory*. It is quite confusing and difficult to comprehend that uncertainty of an event has some relationship with the information gain or loss, just like my mom would never agree on relationship between time and space or just like elementary students of math would find it surprising to learn that gradient and velocity actually refers to the same entity.

In order to understand why *cross entropy* works as a *loss function*, it helps to first understand what entropy serves at the very fundamental level of information and communication and later try to establish a bridge between entropy, information and probability. Finally it would be much easier to fully convince yourself that entropy's ability to quantify the difference between probability distributions made it's way into statistical learning. 
## References
- Elements of information theory/by Thomas M. Cover, Joy A. Thomas.–2nd ed.
- Information theory and evolution / John Avery.
- The information : a history, a theory, a flood / James Gleick.
- [Information theory - Wikipedia](https://en.wikipedia.org/wiki/Information_theory)
