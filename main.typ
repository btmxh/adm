#import "@preview/theorion:0.3.3": *
#import cosmos.rainbow: *
#show: show-theorion

// #set page(height: auto)
#set heading(numbering: "1.1")
#set text(lang: "en")

#set-inherited-levels(1)
#set-zero-fill(true)
#set-leading-zero(true)
#set-theorion-numbering("1.1")

#align(center, strong[meta witch brainrot])

= Differential Privacy

== Motivations

AI algorithms: make decisions from personal data => risk of revealing sensitive
information.

Protected attributes: derivable from seemingly unrelated attributes: someone
eating a lot of Asian food might be Asian, etc.

Conflicting goals: make good predictions (utility) but such predictions should
not be too "personalized" (privacy), or else personal information will be
revealed.

e.g. Netflix challenge: release a huge dataset that is anonymized, though it is
still possible to find out who is who on that dataset... so we can't just remove
protected attributes.

*Differential Privacy*: rigorous mathematical formalization of this

== Definitions

Denote:
- $cal(X)$: domain of all possible inputs of a user.
- $cal(X)^*$: all sequences of inputs

Dataset: $x = (x_1, x_2, ..., x_n) in cal(X)^*$. Each $x_i in cal(X)$ is an user.

Replace $x_i$ in $x$ by another input $x_i'$ yields the dataset $x'$.

Cryptographic security: an algorithm $A$ run on either $x$ or $x'$ produces the
indistinguishible outputs $=>$ zero utility! We can only allow them to be
slightly different to have a non-trivial outcome.

#definition(title: "Neighboring datasets")[
  Neighboring datasets are datasets that differ in exactly one user.
]

#example(title: "Randomized response (Warner '65, privacy)")[
  We want to release the proportion of the population with a certain disease.
  Q&A won't cut it as people don't want to reveal bad things about themselves.
  Instead, the people would response using the following scheme:
  $ R(z) = cases(z "w.p." e^(epsilon)/(e^epsilon + 1), 1 - z "otherwise"), $
  for some small $epsilon > 0$.

  Then, a fixed output $R(z)$ can correspond to either case
  of the inputs,this is called *plausible deniability*. Given two neighboring
  datasets $x$ and $x'$, we can calculate:
  $
    (PP[A(x) = y])/(PP[A(x') = y]) = (product PP[A(x_i) = y_i])/(product
    PP[A(x'_i) = y_i]) = (PP[A(x_k)=y_k])/(PP[A(x'_k)=y_k]) = exp(
      plus.minus
      epsilon
    ) approx 1,
  $
  where $x_k, x'_k$ is the point where the two datasets differ.
] <ex:rng-resp>

#example(title: "Randomized response (Warner '65, utility)")[
  With the scheme described above, we can get an estimator:
  $ V = sum_i (a y_i - b), $
  where $EE[a y_i - b] = x_i$, which means $a 1/(e^epsilon + 1) - b = 0$ and $a
  e^(epsilon)/(e^epsilon + 1) - b = 1$. Solving this yields $a =
  (e^(epsilon)+1)/(e^epsilon-1)$ and $b = 1/(e^epsilon - 1)$.

  The variance of $V$ is,
  #let Var = math.op("Var")
  $
    Var[V] = sum_i Var[a y_i - b] = n a^2 Var[y_i] = n e^epsilon/(e^epsilon -
      1)^2 => sigma = sqrt(n) e^(epsilon/2)/(e^epsilon - 1)
  $
]

Clearly, with the example above, private algorithms can be pretty useful.

#definition(title: "Differential privacy")[
  A (stochastic) algorithm $cal(A): cal(X)^* -> cal(O)$ is
  $epsilon$-differentially private ($epsilon$-DP) if changing
  one person's data does not change the output distribution by much:
  $ (PP[A(x) in S])/(PP[A(x') in S]) <= exp epsilon, $
  for every (measurable) subsets $S$ of the set of all outcomes.
]

In theory, we pick $epsilon approx 0.1$, but in practice, we can let $epsilon
approx 10 div 20$ (utilitymaxxing).

== Laplace mechanism

Use *noise* to protect sensitive information. If we want $f(x)$, we calculate:
$ A(x) = f(x) + "error" $
instead. The error should depend on the "sensitivity" of $f$ (something like the
Lipschitz constant?).

#let GS = "GS"
#definition(title: "Global sensitivity")[
  For $f: cal(X)^* -> RR^d$, the global sensitivity of $f$ is:
  $ GS_f = sup_(x, x' "are neighbors") norm(f(x) - f(x'))_1. $
]

The Laplace mechanism is the following:
#theorem(title: "Laplace mechanism")[
  If $f$ is a vector-valued function with global sensitivity $Delta$, then
  $ A(x) = f(x) + Z, $
  where $Z_i tilde "Lap"(Delta/epsilon)$, is $epsilon$-DP.
]

#proof[
  Let $x$ and $x'$ be neighboring datasets. Pick a vector $y in RR^d$, let $u =
  f(x), v = f(x')$. Denote $h_x$ as the PDF of $A(x)$, we compute:
  $
    (h_x (y))/(h_(x')(y)) = (product_i (e/(2 Delta) exp(
        -epsilon/Delta abs(
          y_i -
          u_i
        )
      )))/(product_i (e/(2 Delta) exp(-epsilon/Delta abs(y_i - v_i)))) =
    exp(epsilon/Delta (norm(y-u)_1 - norm(y-v)_1)) <= exp(
      epsilon/Delta
      norm(u-v)_1
    )<= exp epsilon.
  $
]

Utility: the error $abs(A(x) - f(x))$ is not going to be much because Laplace
distribution is centered pretty nicely:
- $EE[norm(Z)_1] = d lambda,$
- $PP[M >= lambda(ln d + t)] <= exp(-t),$ where $M = norm(Z)_infinity$.

#exercise[Find a bound for $EE[M]$.]

// #solution[
// ]

== Basic composition and post-processing

Given dataset $x$, we perform multiple analyses on $x$. If each analysis is
$epsilon$-DP: is the sensitive data safe?

#example[
  Consider a private $x$ and a random $y$. Given $y$, trivially $x$ can't be
  deduced. Given $x + y$, trivially $x$ can't be deduced. But given $y$ and $x +
  y$, we are cooked.
]

#lemma[
  $A: cal(X)^* -> Y_1 times Y_2$, defined as:
  $ A(x) = (y_1, y_2), y_1 = A_1 (x), y_2 = A_2 (x, y_1), $
  for $epsilon_1$-DP $A_1$ and $epsilon_2$-DP $A_2$ is $(epsilon_1 +
    epsilon_2)$-DP.
]

#proof[
  Take neighboring datasets $x, x'$, outcome $y = (y_1, y_2)$. The rest is
  trivial.
]

#exercise[Verify the plus (formalize it as XOR-ing bits first) example above.]

#lemma(title: "Composition lemma")[
  Given a sequence of (stochastic) algorithms $cal(A)_i$:
  - $cal(A)_1: cal(X)^* -> Y_1$,
  - $cal(A)_i: product_(j =1)^(i-1) Y_j times cal(X)^* -> Y_i,$
  - $cal(A)_i$ (with fixed $y_i$'es) is $epsilon_i$-DP,
  then the composition of all $cal(A)_i$ is $(sum_i epsilon_i)$-DP.
]

#proof[
  Trivial generalization of the above lemma.
]

#lemma(title: "Post-processing lemma")[
  The composition of any arbitrary function and an $epsilon$-DP algorithm is
  $epsilon$-DP.
]

#proof[
  // https://chatgpt.com/s/t_6864d6c6fdb88191bee38448c9678767
  Let $A: cal(X)^* -> cal(Y)$, $B: cal(Y) -> cal(Z)$. Assume that
  $B$ is deterministic (the randomized case can be similarly proven).
  Take neighboring datasets $x, x'$ and any event $S subset.eq cal(Z)$. We
  compute:
  $
    PP[B(A(x)) in S]/PP[B(A(x')) in S] = PP[A(x) in B^(-1) (S)]/PP[A(x') in
      B^(-1) (S)] <= exp epsilon.
  $

  In the randomized $B$ case, if we emulate the randomness of $B$ via a random
  parameter $R$: $B(A(x)) = B_d (A(x), r)$, where $B_d$ is deterministic.
  $ PP[B(A(x)) in S] = integral_S PP[B_d (A(x), r) in S] dif P_R (r). $
  This can be easily bound using the result for deterministic $B$.
]

#exercise[What if $B$ is more private than $A$? If so, can $B compose A$ be more
  private than $A$?]

#theorem(title: "Group privacy theorem")[
  Given datasets $x$ and $x'$ that differ in $k$ positions. If $A$ is
  $epsilon$-DP, then for any event $E$,
  $ PP[A(x) in E]/PP[A(x') in E] <= exp k epsilon. $
]

#proof[
  Trivial (change bit by bit of the datasets).
]

Consider the dataset $x_1, x_2, ..., x_n in RR^d$, where $x_i in cal(U) = {x:
  norm(x)_1 <= 1}$.

#example(title: "DP K-means")[
  K-means find $k$ centers $c_1, ..., c_k$ to minimize the clustering error:
  $ E = sum_(i=1)^n min_(j in [k]) norm(x_i - c_j)_2^2. $

  *Recall*: k-means is the following algorithm:
  ```py
  def kmeans[DataPoint](D: list[DataPoint], K: int):
    centroids = choices(D, K) # or some other init scheme
    while True:
      yield centroids
      S[j] = {i: centroids[i] is closest to D[i]}
      n[j] = len(S[j])
      a[j] = sum(D[i] for i in S[j])
      centroids = [a[j]/n[j] for j in range(K)]
  ```

  Let's make it DP:
  ```py
  def kmeans[DataPoint](D: list[DataPoint], K: int):
    centroids = choices(D, K) # or some other init scheme
    for _ in range(T):
      yield centroids
      S[j] = {i: centroids[i] is closest to D[i]}
      n[j] = len(S[j])
      a[j] = sum(D[i] for i in S[j])
      n[j] += lap(2/epsilon) # Delta of n[j] and a[j] are both 2
      a[j] += lap(2/epsilon)
      centroids = [safe_div(a[j], n[j]) for j in range(K)]
  ```

  This algorithm, by the composition lemma is $2T epsilon$-DP.
]

== Exponential mechanism

#example[
  There are $d$ candidates, everyone can vote for any subset of the candidates.
  The person with the highest vote win.
  If we Release all the counts. Then, the noise wrt Laplace mechanism should be
  $"Lap"(d/epsilon)$. With constant probability, the person with the highest
  (noisy) count has at least the max true count $- (d ln d)/epsilon$.
]

#definition(title: "Exponential mechanism")[
  Given a set $cal(Y)$ of possible outcomes, a score function $q: cal(Y) times
  cal(X)^* -> RR$. Sensitivity:
  $
    Delta = sup_(y in cal(Y)) sup_(x, x' "are neighbors") abs(
      q(y, x) - q'(y,
        x)
    ).
  $

  The *exponential mechanism* would select $Y$ from the distribution $ PP[Y = y] prop exp (epsilon/(2 Delta) q(y, x)). $
]

Returning to the problem above. Define:
$ q(y, x) = \#("votes for" y "in" x), $
then $Delta = 1$.

If the counts are $d/epsilon, 0, 0, ..., 0$, then:
$ PP[Y = 1] = exp(d/2)/(exp(d/2) + d - 1). $

#theorem[
  Exponential mechanism is $epsilon$-DP.
]

#proof[
  Considering neighboring datasets $x_1, x_2$ and an outcome $y$:
  $
    PP[Y=y|x = x_1]/PP[Y=y|x=x_2] & = (exp(epsilon/(2 Delta) q(y, x_1)) sum_(y')
                                    exp(epsilon/(2 Delta) q(y', x_2)))/(exp(epsilon/(2 Delta) q(y, x_2)) sum_(y')
                                    exp(epsilon/(2 Delta) q(y', x_1))) \
                                  & <= exp(epsilon/2) exp(epsilon/2) = exp
                                    epsilon.
  $
]

#theorem[
  The output of exponential mechanism $y$ would satisfy:
  $ PP[q(y, x) <= q_"max" - (2 Delta (ln abs(Y) + t))/ epsilon] <= exp(-t). $
  where $q_"max" = max_y q(y, x)$.

  As a consequence, $EE_y [q(y, x)] >= q_"max" - (2 Delta (ln abs(Y) + 1))/
  epsilon$.
]

#proof[
  Let $B$ be the set of bad outcomes: the set of all $y$ such that
  $ q(y, x) <= q_"max" - (2 Delta (ln abs(Y) + t))/ epsilon. $

  Then, for every $y in B$
  $
    PP[Y in B] & = (sum_(y in B) exp(Delta/(2 epsilon) q(y, x)))/
                 (sum_y' exp(Delta/(2 epsilon) q(y', x)))              \
               & <= (abs(B) exp(
                   Delta/(2 epsilon) (q_"max" - ((2 Delta)/epsilon (ln abs(Y) +
                         t)))
                 ))/(sum_y' exp(Delta/(2 epsilon) q(y', x)))           \
               & <= (abs(B) exp ((Delta)/(2 epsilon) q_"max" - ln abs(Y) - t)
                 )/(abs(Y) exp(Delta/(2 epsilon) q_"max")) <= exp(-t).
  $
]

= Principled DL and NLP

== Supervised Learning

3 main problems:
- Approximation: How good can functions in $cal(H)$ approximate the oracle?
- Optimization: How to find a function in $cal(H)$ that approximate the oracle
  well for a given dataset?
- Generalization: How good is the found solution at generalizing?

GLBM: map input using a predefined non-linear mapping to introduce non-linearity
into the system. Basis function: polynomials, gaussian, sigmoid.

= Optimal Transport in Large-scale ML/DL

== Motivation

Given dataset $X_1, X_2, ..., X_n in RR^d$. Belief: $X_i tilde P$ for some
distribution $P$. We want to learn $P$.

Basic approach: Empirical distribution:
$ P_n = 1/n sum_i delta_(x_i). $
Then,
$ D(P_n, P) <= c n^(-1/d), $
given a metric $D$. $c$ is a universal constant. *Curse of dimensionality*: if
we want the error to be smaller than a given constant, the sample size needs
to increase exponentially as $d$ increases.

Big data: weak signal to noise, e.g. finance, robot, medical, etc.
- Finance: low signal to noise
- Medical: no risk management solution => conformal prediction (needs lot of
  maths!)
- Robot: physics

To solve the smoothness problem, we can smoothen each Dirac delta using a
Gaussian:
$
  f_sigma = P_n convolve K_sigma, K_sigma (x) = 1/(sigma sqrt(2pi)) exp
  (-norm(x)_2^2/(2 sigma^2)),
$
where $sigma$ is a smoothing parameter.

Then,
$ norm(P_n convolve K_sigma - P) <= C ( sigma^(c(d)) + n^(-1/d)) -> 0, $
as $sigma -> 0, n -> infinity$.

Introducing Deep Learning:
- 2011: VAE (beyond images)
- 2014: GAN
- 2018: Diffusion models (slow)

=== GAN

Starting from a Gaussian $X tilde cal(N)$, we find a transform $T$ from $cal(N)$
to $P$:
$ min D(T \# cal(N), P) $

DM: add noises to data to turn it into Gaussian (fwd step), then denoise to turn
it into generated data.

$D$: KL-divergence, JS-divergence, *optimal transport*.
