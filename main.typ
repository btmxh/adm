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

#let ip(x, y) = $lr(angle.l #x, #y angle.r)$

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

== Differential Privacy

=== Definitions

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

=== Laplace mechanism

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

=== Basic composition and post-processing

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

=== Exponential mechanism

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
    PP[Y in B] & = (sum_(y in B) exp(epsilon/(2 Delta) q(y, x)))/
                 (sum_y' exp(epsilon/(2 Delta) q(y', x)))        \
               & <= (abs(B) exp(
                   epsilon/(2 Delta) (q_"max" - ((2 Delta)/epsilon (ln abs(Y) +
                         t)))
                 ))/(sum_y' exp(epsilon/(2 Delta) q(y', x)))     \
               & <= (abs(B) exp (epsilon/(2 Delta) q_"max" - ln abs(Y) - t)
                 )/( exp(epsilon/(2 Delta) q_"max")) <= exp(-t).
  $

  Then, if $g(t) = 2 Delta/epsilon (ln abs(Y) + t)$ and $Z = q_"max" - q(y, x)
  >= 0$,
  $
    EE[Z] = integral_0^infinity z f_Z (z) dif z = integral_0^infinity
    integral_0^z f_Z (z) dif t dif z = integral_0^infinity underbrace(
      integral_t^infinity
      f_Z (z) dif z, #[$PP[Z >= t]$]
    ) dif t = integral_0^infinity PP(Z >= z) dif z
  $
  Using this:
  $
    EE[Z] & = integral_0^infinity PP(q(y, x) <= q_"max" - z) dif z             \
          & = integral_0^infinity PP(q(y, x) <= q_"max" - g(t)) dif (z = g(t)) \
          & <= integral_0^infinity exp(-t) g'(t) dif t                         \
          & = underbrace([-exp(-t) g(t)]_0^infinity, 0) + integral_0^infinity
            exp(-t) g(t) dif t = EE[g(T)],
  $
  where $T tilde "Exp"(1)$.

  Since $g$ is an affine function, we can apply "Jensen's equality":
  $ EE[g(T)] = g(EE[T]) = g(1) = (2Delta)/epsilon (ln abs(Y) + 1). $
]

This logarithmic error is much better than the linear error guaranteed by
Laplace mechanism.

*Noisy-max mechanism*: For each outcome $y in cal(Y)$, sample
$ z_y prop "Exp"(2Delta/epsilon) $
#let argmax = math.op("argmax", limits: true)
and return $ argmax_(y in cal(Y)) (q(y, x) + z_y). $
This achieves a similar result as the exponential mechanism.

If we replace $"Exp"$ by the Gumbel distribution, we get *the identical result* as
exponential mechanism.

If we want to take the top-$k$ highest outcomes, you can run exponential
mechanism $k$ times. By the composition theorem, this is $k epsilon$-DP.

=== Binary tree mechanism

Given a dataset $x_1, x_2, ...$ that arrives *sequentially*. We want to make queries:
$ f_(s, t) (x) = sum_(i = s)^t phi(x_i). $
Assuming $phi(x_i) <= 1$, if we change $phi(x_i)$ of one user, at most
$cal(Theta)(n^2)$ queries change, so the amount of Laplace noise we need to add
is $"Lap"(Theta(n^2)/epsilon)$. The magnitude of noise is quadratic, while
the query values should grow linearly at most, which is problematic!

Instead, we use a new set of queries: $Q = {f_(2^i j + 1, 2^i (j + 1)): i, j in
  NN}$. Then,
$abs(Q) = 2n-1$, and the global sensitivity is only $log_2 n + 1 = Theta(log_2 n)$.

Every original query can be answered by $Theta(log_2 n)$ queries from $Q$. This
is basically what segment trees in CP do, I'm not elaborating this here.

#theorem(title: "Binary tree mechanism error")[
  Let $a_(s, t) = f_(s, t) + "Lap"((log_2 n + 1)/epsilon)$ for all $f_(s, t) in
  Q$.
  Then, $ EE[max_(s, t) abs(a_(s, t) - f_(s, t))] <=Theta((log n)^3/epsilon). $
]

#proof[
  For $(s, t) in I = {(s, t)$ such that $f_(s, t) in Q$, we have:
  $
    PP[max_((s, t) in I) abs(a_(s, t) - f_(s, t)) >= (log_2 n + 1)/epsilon
      (ln(2n-1) + t)] <= exp(-t),
  $
  and therefore,
  $
    EE[max_((s, t) in I) abs(a_(s, t) - f_(s, t))] <= (log_2 n + 1)/epsilon (ln
      (2n-1) + 1) = Theta((log n)^2/epsilon).
  $
  For every $(s, t)$ (not just in $I$), the error can be accumulated from at
  most $Theta(log_2 n)$ sub-errors, so
  $
    EE[max_(s, t) abs(a_(s, t) - f_(s, t))] <= Theta((log n)^2/epsilon) times
    Theta(log_2 n) = Theta((log n)^3/epsilon).
  $
]

*Problem:* What is $n$? Our data can come infinitely! Alternatively, we can use
non-uniform error: error on layer $n$ is $epsilon/n^2$, so the sum of
$epsilon$'es is bounded.

== Approximate Differential Privacy

=== Definitions

Approximate Differential Privacy is a generalization of "pure" Differential
Privacy, which is defined as follows:
#definition(title: "Approximate Differential Privacy")[
  A (stochastic) algorithm $A: cal(X)^* -> cal(Y)$ is $(epsilon, delta)$-DP if
  for any neighboring datasets $x, x'$ and event $E subset.eq cal(Y)$,
  $ PP[A(x) in E] <= PP[A(x') in E] exp (epsilon) + delta. $

  Pure differential privacy is the special case when $delta = 0$.
]

Approximate DP also satisfies basic properties that pure DP do:

#lemma(title: "Post-processing lemma")[
  The composition of any arbitrary function and an $(epsilon, delta)$-DP
  algorithm is also $(epsilon, delta)$-DP.
]

#proof[
  We have:
  $ PP[B(A(x)) in S] = integral_R PP[B_d (A(x), r) in S] dif PP_R (r), $
  where the function $B$ is decomposed into a deterministic form $B_d (dot, r)$.
  Then, for any neighboring datasets $x$ and $x'$,
  $
    PP[B(A(x)) in S] & = integral_R PP[B_d (A(x), r) in S] dif PP_R (r)                       \
                     & =
                       integral_R PP[(A(x), r) in B_d^(-1) (S)] dif PP_R (r)                  \
                     & = integral_R PP[A(x) in [B_d^(-1) (S)]_r] dif PP_R (r)                 \
                     & <= integral_R (PP[A(x') in [B_d^(-1) (S)]_r] exp(epsilon) + delta) dif PP_R
                       (r)                                                                    \
                     & = exp(epsilon) integral_R PP[B_d (A(x'), r) in S] dif PP_R (r) + delta \
                     & = exp(epsilon) PP[B(A(x')) in S] + delta.
  $
  Here, $[E]_r$ denotes the cross section of the set $E$ given the parameter
  $r$: $E subset.eq cal(Y) times R => [E]_r = {y in cal(Y): (y, r) in E}$.
]

#lemma(title: "Composition lemma")[
  Given a stochastic algorithm $A: X^* -> Y_1 times Y_2$, defined as:
  $ A(x) = (y_1, y_2), y_1 = A_1 (x), y_2 = A_2 (x, y_1), $
  for $(epsilon_1, delta_1)$-DP $A_1$ and $(epsilon_2, delta_2)$-DP $A_2$. Then,
  $A$ is $(epsilon_1+epsilon_2, delta_1 + delta_2)$-DP.
]

#proof[
  The proof of this is somewhat technical, see Appendix B of The Algorithmic
  Foundations of Differential Privacy.
]

#let good = "Good"
#let bad = "Bad"
#lemma[
  Consider two neighboring datasets $x, x'$ and an algorithm $A$.
  Denote:
  $
    good_(x, x') = {y in Y: PP[A(x) = y]/PP[A(x')=y] <= exp epsilon}, bad_(x, x')
    = Y without good_(x, x').
  $
  Then, if $PP[A(x) in bad_(x, x')] <= delta$ then $A$ is $(epsilon, delta)$-DP.
] <lem:strong-adp>

#proof[
  For any event $E$,
  $
    PP[A(x) in E] & = PP[A(x) in E sect good_(x, x')] + PP[A(x) in E sect bad_(x,
                      x')]                                                    \
                  & <= exp (epsilon) PP[A(x') in E sect good_(x, x')] + delta \
                  & <= exp (epsilon) PP[A(x') in E] + delta.
  $
]

=== Examples

#theorem(title: "Truncated Laplace mechanism")[
  Replace the noise distribution of Laplace mechanism by a Truncated Laplace
  distribution $"Lap"(lambda, tau)$:
  $
    "PDF"_(lambda, tau) (y) prop cases(
      exp(-abs(y)/lambda) "if" abs(y) <= tau, 0
      "otherwise"
    ).
  $
  This
  gives the Truncated Laplace mechanism, which is $(epsilon, delta)$-DP when
  $tau = cal(O) (log (1/delta))$:
  $ A(x) = f(x) + Delta/epsilon "Lap"(1, tau). $
]

#proof[
  Denote the PDF of the error as $k(x)$, $z_tau = integral_(-tau)^tau
  exp(-abs(y)) dif y = 2(1-exp(-tau))$ then:
  $ k(y) = cases(1/z_tau exp(-abs(y)) "if" abs(y) <= tau, 0 "otherwise"). $
  Let $x, x'$ be neighboring datasets. Let $f: X^* -> RR^d$, and pick $y in
  RR^d$. Denote $u = f(x), v = f(x')$, $h_x$ as the PDF of $A(x)$, then:
  $
    h_x (y) = PP[A(x) = y] = PP["error" = epsilon/Delta (y - f(x))] =
    k(epsilon/Delta (y - u)),
  $
  and
  $ h_(x') (y) = k(epsilon/Delta (y - v)). $

  If $abs(epsilon/Delta (y - u)) > tau$, then the desired result is trivially
  true. Assuming otherwise, then:
  $ h_x (y) = k(epsilon/Delta (y - u)) = 1/z_(tau) exp(-(epsilon)/( Delta) abs(y-u)) $, so there are two cases:
  - If $abs(epsilon/Delta (y - v)) > tau$, then our desired result holds only
    when $h_x (y) < delta$. Hence, we need
    $ delta > 1/z_tau exp(- epsilon/Delta abs(y - u)). $
    Now, since $abs(y-v) > tau Delta/epsilon$, we must have:
    $abs(y-u) >= abs(y-v) - abs(u-v) <= Delta (tau/epsilon - 1)$.
    Hence,
    $
      1/z_tau exp(-epsilon/Delta abs(y-u)) <= 1/z_tau exp(
        -epsilon/Delta dot Delta
        (tau/epsilon - 1)
      ) = 1/z_tau exp (epsilon-tau) = exp(epsilon)/(2(exp(tau)-1)).
    $
    We need $delta > exp(epsilon)/(2(exp tau - 1)) <=> tau >
    ln(1+exp(epsilon)/(2 delta))$, true when $tau = cal(O)(log 1/delta)$.
  - If $abs(epsilon/Delta (y-v)) <= tau$, then
    $
      (h_x (y)) / (h_(x') (y)) = (exp(
        -epsilon/Delta
        abs(y-u)
      ))/(exp(-epsilon/Delta abs(y-v))) = exp(
        epsilon/Delta
        (abs(y-v)-abs(y-u))
      ) <= exp(-epsilon/Delta abs(u-v)) = exp(epsilon).
    $
    Hence, $h_x (y) <= exp(epsilon) h_(x') (y) <= exp(epsilon) h_(x') (y) +
    delta$.
]

#theorem(title: "Gaussian mechanism")[
  The Gaussian mechanism is:
  $ A(x) = f(x) + cal(N) (0, (2 Delta^2 ln(2/delta))/epsilon^2) $
  For $epsilon <= 1$ and $delta > 0$, the Gaussian mechanism is $(epsilon,
    delta)$-DP.
]

#proof[
  We use @lem:strong-adp. Fix two neighboring datasets $x, x'$.
  $
    y in bad_(x, x') & <=> ln (f_(A(x)) (y))/(f_(A(x')) (y)) > epsilon,
  $
  where
  $
    epsilon < ln(f_(A(x)) (y))/(f_(A(x')) (y)) & = ((y-f(x'))^2-(y-f(x))^2)/(2 sigma^2)     \
                                               & =
                                                 ((f(x)-f(x'))(2y-f(x)-f(x')))/(2sigma^2)   \
                                               & <= (Delta (Delta + 2 abs(
                                                     y -
                                                     f(x)
                                                   )))/(4 Delta^2 ln (2/delta) 1/epsilon^2) \
                                               & = (epsilon^2 (Delta +
                                                   2abs(y-f(x))))/(4 Delta ln (2/delta)).
  $
  Solving for $abs(y-f(x))$ yields,
  $ abs(y-f(x)) >= (sqrt(2) Delta)/epsilon ln(2/delta). $
  However, the tail bound of Gaussian implies that:
  $ PP[abs(y-f(x)) >= (sqrt(2) Delta)/epsilon ln(2/delta)] <= delta, $
  which implies that $PP[y in bad_(x, x')] <= delta$.
]

A trivial generalization of this theorem is to multivariate normal error.
#theorem(title: "Multivariate Gaussian mechanism")[
  If $f: X^* -> RR^k$, then
  $ A(x) = f(x) + bold(cal(N))(bold(mu), sigma^2 bold(I)_(k times k)), $
  with $sigma^2 = (2 Delta_2^2 ln(2/delta))/epsilon^2$ and $bold(mu) = 0$ is
  $(epsilon, delta)$-DP, where $Delta_2$ is the $cal(l)_2$ sensitivity of $f$:
  $ Delta_2 = sup_(x, x' "are neighbors") norm(f(x)-f(x'))_2. $
]

#corollary[
  The Multivariate Gaussian mechanism gives
  $
    EE[max_(j in [k]) abs(f_j (x)-a_j)] <= cal(O) (sqrt(
        k ln k
        ln(1/delta)
      )/epsilon).
  $
]

For reference, this bound for the Laplace mechanism is $cal(O)((k ln
  k)/epsilon)$.

=== Advanced composition

Basic composition theorem states that composing $k$ $(epsilon, delta)$-DP
algorithms gives us a $(k epsilon, k delta)$-DP algorithm. However, this bound
can be made tighter (w.r.t. $epsilon$).

#theorem(title: "Advanced composition theorem")[
  For $epsilon, delta > 0$ and $delta' > 0$, the composition of $k$ mechanisms,
  each of which is $(epsilon, delta)$-DP, is $(tilde(epsilon), tilde(delta))$-DP,
  where:
  $
    tilde(epsilon) = epsilon sqrt(2k ln(1/delta')) + k epsilon (e^epsilon -
    1)/(e^epsilon + 1), tilde(delta) = k delta + delta'.
  $
]

#example[
  If we pick $delta' = delta$, then:
  $ tilde(epsilon) = epsilon sqrt(2k ln(1/delta)) + O(k epsilon^2), $
  where $(e^epsilon-1)/(e^epsilon+1) approx epsilon / 2$.
  If we want the final mechanism to have $tilde(epsilon) < 1$, then we pick
  $epsilon approx 1/(sqrt(k ln (1/delta)))$.
  Clearly, these results are much stronger than pure DP, at the cost of having
  an uncertainty amount of $delta$.

  The $delta'$ term serves as a trade-off factor between $tilde(epsilon)$ and
  $tilde(delta)$.
]

#proof[
  Model the privacy loss as a random variable:
  $ I_(x, x') (y) := ln (PP[A(x) = y])/(PP[A(x') = y]). $

  By @lem:strong-adp, if we can show:
  $ PP_(y tilde A(x)) [I_(x, x') (y) > epsilon] <= delta, $
  then $A$ is $(epsilon, delta)$-DP.

  When $A$ is a composition of $A_1, ..., A_k$, which returns a sequence $y =
  (y_1, y_2, ..., y_k)$:
  $
    PP[A(x) = (y_1, ..., y_k)] & = PP[A_1 (x) = y_1] PP[A_2 (x, y_1) = y_2] ...
                                 PP[A_k (x, y_1, ..., y_(k-1)) = y_k]                        \
                               & = product_(n = 1)^k PP[A_n (x, y_1, ..., y_(n - 1)) = y_n].
  $
  Then,
  $
    I_(x, x') (y) = sum_(n = 1)^k underbrace(
      ln (PP[A_n (x, y_1, ..., y_(n - 1)) =
        y_n])/(PP[A_n (x', y_1, ..., y_(n - 1)) = y_n]), X_n
    ).
  $
  Each $X_n$ is a random variable which is in the range $[-epsilon, epsilon]$
  most of the time (except for probability $delta$).

  Intuition:
  - Since $X_n in [-epsilon, epsilon]$ most of the time, the variance is about
    $epsilon^2$, so the variance of the sum is about $k epsilon^2$, which gives
    the $epsilon sqrt(k)$ term in the upper bound.
  - The mean of $X_n$ is about $epsilon^2$, so the total mean is about $k
    epsilon^2$, which gives rise to the $k epsilon (e^epsilon - 1)/(e^epsilon^+
    1)$ term.
]

Now, we will look at some privacy losses to get some more intuition.
- Gaussian mechanism: $I_(x, x')(y) tilde cal(N) (Delta_2^2/(2 sigma^2),
    Delta_2^2/sigma^2)$ ($Delta_2$: $cal(l)_2$ sensitivity, $sigma^2$: variance
  of the noise), where $sigma = Delta/epsilon sqrt(2 ln (2/delta))$.
  The expected privacy loss is:
  $ EE[I_(x, x') (y)] = epsilon^2/(4 ln (2/delta)) approx epsilon^2. $
- Randomized response:
  $ y_i = cases(x_i "w.p." e^epsilon/(e^epsilon+1), 1-x_i "otherwise"). $
  Calculating $I_(x, x')$:
  - $I_(0, 1) (0) = ln((PP[A(0) = 0])/(PP[A(1)=0])) = epsilon,$
  - $I_(0, 1) (1) = ln((PP[A(0) = 1])/(PP[A(1)=1])) = -epsilon,$
  so the expected privacy loss $EE_(y tilde A(0)) [I_(0, 1) (y)] = epsilon
  e^epsilon/(e^epsilon + 1) - epsilon 1/(e^epsilon + 1) = epsilon (e^epsilon -
  1)/(e^epsilon + 1) approx epsilon^2.$

#problem[Calculate the expected privacy loss for Laplace mechanism.]

We continue with the proof:

Given $X$ and $Y$ are random variables, then we write:
$
  X approx_(epsilon, delta) Y <=> PP[X in E] <= exp(epsilon PP[Y in E]) +
  delta, forall "event" E.
$

Then,
$ A_1 (x) approx_(epsilon, delta) A_1 (x'), ... $

Consider two special RVs $U$ and $V$, we will attempt to prove the theorem for
these random variables:
#table(
  columns: (1fr, 1fr, 1fr),
  align: center,
  table.header()[Outcome][$P_u$][$P_v$],
  [0], $e^epsilon/(e^epsilon+1) (1 - delta)$, $(1-delta)/(e^epsilon+1)$,
  [1], $(1-delta)/(e^epsilon+1)$, $e^epsilon/(e^epsilon+1) (1 - delta)$,
  [I am U], $delta$, [0],
  [I am V], $0$, $delta$,
)

#lemma(title: "Simulation Lemma")[
  For any pair of RV $X approx_(epsilon, delta) Y$, there exists a randomized
  mapping $F$ such that:
  $ F(U) tilde X, F(V) tilde Y. $
]

Basically this lemma states that, every pair of $(X, Y)$ is basically $U$ and
$V$ before a post-processing step. Note that since post-processing preserves
privacy, this does not change anything at all.

#proof[
  - If $delta = 0$, then for every outcome $z$, let $p_X (z), p_Y(z)$ be the
    probability of outputing $z$ for the distributions $X$ and $Y$,
    respectively.

    Since $F(U) tilde X$, $p_X (z) = PP[F(U) = z] = e^epsilon/(e^epsilon + 1)
    PP[F(0)=z] + 1/(e^epsilon + 1)PP[F(1)=z].$
    Similarly,
    $p_Y (z) = e^epsilon/(e^epsilon+1) PP[F(1) = z] + 1/(e^epsilon+1) PP[F(0) =
      z].$ Solving for $PP[F(0)=z]$ and $PP[F(1)=z]$ gives the distribution of
    $F$ given inputs $0$ or $1$:
    - $PP[F(0)=z] = (e^epsilon p_X (z) - p_Y (z))/(e^epsilon + 1),$
    - $PP[F(1)=z] = (e^epsilon p_Y (z) - p_X (z))/(e^epsilon + 1).$
    This satisfies non-negative due to $epsilon$-DP, and the total sum adds up
    to 1, so this is a probability distribution.
  - If $delta != 0$, then $PP[F(0)=z]$ and $PP[F(1)=z]$ can be negative for some
    $z$ (since we don't have $epsilon$-DP). In other words, we need to handle
    the region $A = {z: p_X (z) > e^epsilon p_Y (z)}$ and $B = {z: p_Y (z) >
      e^epsilon p_X (z)}$.
    Then, we can define:
    - $PP[F(0)=z] = (e^epsilon p_X (z) - p_Y (z))/(e^epsilon + 1),$ if $z in.not
      A$,
    - $PP[F(1)=z] = (e^epsilon p_Y (z) - p_X (z))/(e^epsilon + 1),$ if $z in.not
      B$.
    - $PP[F("I am U") = z] = ...$
    - $PP[F("I am V") = z] = ...$
]

#exercise[
  $P = "Lap"(1/epsilon), Q = 1 + "Lap"(1/epsilon)$. Give the randomized
  $F$ such that $F(U) tilde P, F(V) tilde Q$.
]

#solution[
  Denote $f_X$ as the PDF of $X$.
  Assuming $delta = 0$.
  The PDF of $F(U)$ is:
  $
            epsilon/2 exp(-epsilon abs(x)) = f_F(U) (x) & = f_F(0) (x) f_U (0) + f_F(1)
                                                          (x) f_U (1) \
                                                        & = (f_F(0) (x) exp epsilon +
                                                          f_F(1) (x) )/(exp epsilon +
                                                          1)          \
    => epsilon/2 exp(-epsilon abs(x)) (exp epsilon + 1) & = f_F(0) (x) exp epsilon +
                                                          f_F(1) (x).
  $
  The PDF of $F(V)$ is:
  $
            epsilon/2 exp(-epsilon abs(x-1)) = f_F(V) (x) & = f_F(0) (x) f_U (0) + f_F(1)
                                                            (x) f_U (1)             \
                                                          & = (f_F(0) (x) + f_F(1) (x)
                                                            exp epsilon )/(exp epsilon
                                                            + 1).                   \
    => epsilon/2 exp(-epsilon abs(x-1)) (exp epsilon + 1) & = f_F(0) (x) +
                                                            f_F(1) (x)exp epsilon .
  $
  Solving yields:
  $
    mat(exp epsilon, 1; 1, exp epsilon) vec(f_F(0) (x), f_F(1) (x)) =
    epsilon/2
    (exp(epsilon)+1)
    vec(exp(-epsilon abs(x)), exp(-epsilon abs(x-1)))\
    => vec(f_F(0) (x), f_F(1) (x)) = underbrace(
      epsilon/(2(exp epsilon - 1)), C
    ) mat(exp epsilon, -1; -1, exp epsilon) vec(
      exp(-epsilon abs(x)),
      exp(-epsilon abs(x-1))
    ).
  $
  Hence,
  $
    f_F(0) (x) = C(exp (epsilon - epsilon abs(x)) - exp(-epsilon abs(x - 1)) ),\
    f_F(1) (x) = C(exp(epsilon - epsilon abs(x - 1)) - exp(-epsilon abs(x))).
  $
  Alternatively,
  $
    f_F(0) (x) = epsilon/(2(exp epsilon - 1)) dot cases(
      exp(epsilon (x + 1)) - exp(epsilon (x - 1))
      "if" x <= 0,
      exp(epsilon(1-x) - exp(epsilon(x-1))) "if" 0 < x <= 1,
      0 "otherwise,"
    )\
    f_F(1) (x) = f_F(0) (1 - x).
  $
]

Returning to the theorem.

#proof[
  Since $A_n (x, y_1, ..., y_(n - 1)) approx_(epsilon, delta) A_n (x', y_1, ...,
    y_(n - 1))$, there exists a randomized mapping $F_(y_1, ..., y_(n - 1))$
  on $U$ and $V$ such that:
  $
    F_(y_1, ..., y_(n - 1)) (U) tilde A_n (x, y_1, ..., y_(n - 1)), "and"\
    F_(y_1, ..., y_(n - 1)) (V) tilde A_n (x', y_1, ..., y_(n - 1)).
  $

  Then, we can construct a randomized $F^*$ such that:
  $
    F^* (U_1, ..., U_k) tilde A(x): U_1, ..., U_k tilde_"i.i.d." U,\
    F^* (V_1, ..., V_k) tilde A(x'): V_1, ..., V_k tilde_"i.i.d." V.
  $

  Here is a rough pseudocode version of $F^*$:
  ```py
  def F_star(Us):
    for i in range(1, k + 1):
      y[i] = F_(y[1], y[2], ..., y[i - 1])(U[i])
    return y[1], y[2], ..., y[k]
  ```

  With this, we reduce the problem to the simplest case on $U_1, ..., U_k$ and
  $V_1, ..., V_k$. Since DP is closed under post-processing, we can skip the
  existence of $F^*$ entirely.

  We need to prove:
  $ U = (U_1, ..., U_k) approx_(tilde(epsilon), tilde(delta)) V = (V_1, ..., V_k), $
  where $tilde(epsilon) = epsilon sqrt(2k ln(1/delta')) + k epsilon (e^epsilon -
  1)/(e^epsilon + 1), tilde(delta) = k delta + delta'$.

  Let $y$ be the outcome of $U$ (a $k$-dimensional vector with components either
  0, 1, or I am U).
  - If $y$ contains bad outcomes (I am U or I am V), then:
    $ PP[U "has 'I am U'" or V "has 'I am V'"] = 1 - (1 - delta)^k <= delta k. $
    Denote
  - Otherwise, $y$ is a bit vector (an element of ${0, 1}^k$):
    $ ln (p_U (y))/(p_V (y)) = sum_(i = 1)^k ln (p_U_i (y_i))/(p_V_i (y_i)) $
    If $z_i = 0$, then $ln (p_U_i (y_i))/(p_V_i (y_i)) = epsilon$, otherwise,
    $ln (p_U_i (y_i))/(p_V_i (y_i)) = -epsilon$. In general, the privacy loss in
    each step is $epsilon(1-2y_i)$.
    The total loss is:
    $ L = epsilon(k - 2 sum_(i = 1)^n y_i). $
    Note that $EE[epsilon(1-2y_i)] = epsilon (e^epsilon - 1)/(e^epsilon + 1),$
    so the expected loss is $k epsilon (e^epsilon - 1)/(e^epsilon + 1)$.

    By Azuma's inequality, we have:
    $ PP[L > EE[L] + t epsilon sqrt(k)] <= exp(-t^2/2). $
    Pick $t = sqrt(2 ln (1/delta'))$, then:
    $ PP[L > EE[L] + epsilon sqrt(2 k ln(1/delta'))] <= ln(1/delta'). $

  Denote $B_1$ as the event $U$ has 'I am U', $B_2$ as the event $L > EE[L] +
  epsilon sqrt(2 k ln(1/delta'))$, then:
  $ PP[B_1] <= delta k, PP[B_2] <= delta'. $
  Hence, when neither $B_1$ nor $B_2$ happens,
  $
    ln (p_U (y))/(p_V (y)) = L <= underbrace(
      EE[L] + epsilon sqrt(
        2 k
        ln(1/delta')
      ), tilde(epsilon)
    ),
  $
  which is true with probability at least $1 - underbrace(
    (delta k + delta'),
    tilde(delta)
  )$.
]

=== Sparse vector technique

Given a threshold $T$ in advance and a sequence of $m$ adaptive queries $q_1,
q_2, ..., q_m$. Each query $q_k$ has sensitivity $1$. We want to report the
first $c$ queries that have output exceeding the threshold $T$.

#example[
  Given medical records of $N$ people and diseases $D_1, D_2, ..., D_m$. We want
  to find the first $c = 3$ diseases affecting more than $T = 100$ people. It
  could be $D_1, D_2, D_3$, or $D_1, D_2, D_4$, etc.
]

We first focus on the special case $c = 1$. The sparse vector technique is the
following algorithm.

```py
def above_threshold(D: Dataset, Qs: list[Query], T, epsilon):
  T_hat = T + Lap(2/epsilon)
  indices = []
  for i, query in enumerate(Qs):
    v_i = Lap(4/epsilon)
    if query(D) + v_i >= T_hat:
      indices.append(i)
      break
  return indices
```

In DP texts, the output of `above_threshold` is typically a vector $v$ such
that:
$ v_i = bold(1)[i in "indices"]. $
For the general case ($c$ can be greater than 1 but $c << M$),
this is a *sparse vector*, hence the name.

#theorem[
  Sparse vector technique (`above_threshold` defined above) is $epsilon$-DP.
]

#proof[
  Denote $A(x)$ as `above_threshold(x)`, $L = "Lap"(2/epsilon), E =
  "Lap"(4/epsilon)$.
  Consider neighboring datasets $x, x'$ and an outcome $y$.
  - If $y$ is empty, then:
    $
      PP[A(x) = y] & = integral_RR product_(k = 1)^m PP[q_k (x) + v_k < hat(T)] dif
                     f_hat(T) (hat(T)) \
                   & = integral_RR dif f_L (L) integral_RR^(m)
                     product_(k=1)^m PP[q_k (x) + v_k < T + L] dif f_E^(m)
                     (v_1, ..., v_m)   \
                   & =
    $
  - If $y$ has an index $k$, then conditioning over $v_1, v_2, ..., v_(k-1)$. We
    have:
    $
      PP[A(x) = y] & = PP[and.big_(i = 1)^(k-1) (q_i (x) + v_i < hat(T)) and q_k (x) +
                       v_k > hat(T)]                                                                                  \
                   & = PP[hat(T) in (underbrace(
                           max_(i < k) (q_i (x) + v_i),
                           g(x)
                         ), q_k (x) +
                         v_k]]                                                                                        \
                   & = integral_RR integral_RR bold(1) [t in (g(x), q_k (x) + v]] f_v_k (v) f_hat(T) (t) dif v dif t.
    $
    Change of variables:
    - $v = v' + g(x) - g(x') + q_k (x') - q_k (x),$
    - $t = t + g(x) - g(x')$, then:
    $
      PP[A(x)=y] = integral_RR integral_RR bold(1) [t' in (g(x'), q'_k (x) +
          v')]] f_v_k (v) f_hat(T) (t) dif v dif t.
    $
    Now, $abs(t - t') = abs(g(x) - g(x')) <= 1$, $abs(v - v') <= abs(
      g(x)
      -g(x')
    ) + abs(q_k (x') - q_k (x)) <= 2$ due to $q_i$ sensitivity being 1.

    Hence, $(f_v_k (v) f_T (t))/(f_v_k (v') f_T (t')) <= exp(epsilon)$, and
    therefore,
    $
      PP[A(x)=y] & <= exp(epsilon) integral_RR integral_RR bold(1) [t' in (g(x'),
                       q'_k (x) + v')]] f_v_k (v') f_hat(T) (t') dif v dif t \
                 & = exp(epsilon) PP[A(x) = y'].
    $
  - If $y$ is empty, then the proof can proceed similarly (the condition $t in
    (g(x), q_k (x) + v_k)$ is replaced by $t in (g(x), infinity)$).
]

#theorem(title: "Accuracy of Sparse vector technique")[
  Given a dataset $x$. Let $I$ denote the output of `above_threshold` for input
  $x$. Then, if $alpha = cal(O)((log m + log (2/beta))/epsilon)$, then with
  probability $1 - c beta$, the following must hold:
  - For all $i in.not I$, $q_i (x) <= T + alpha$.
  - If $i in I$, then $q_i (x) >= T - alpha$.
] <thr:acc-spt>

#proof[
  We prove the theorem for the $c = 1$ case, since `above_threshold` hasn't been
  defined for $c != 1$.

  With probability $beta/2$,
  $ abs(T - hat(T)) <= 2 log(2/beta)/epsilon. $
  Similarly,
  $ PP[max_(i in [m]) abs(v_i) >= 4/epsilon (log m + t)] < exp(-t), $
  then we simply choose $t = log(beta/2)$.
]

Now, we can finally define `above_threshold` for the $c > 1$ case:
We simply repeat the algorithm $c$ times.
The result of @thr:acc-spt can be proven trivially.

== Private Gradient Descent

Consider the minimization problem:
$ min_(x in C) f(x), $
then, Projected GD attempts to solve this problem as follows:
```py
def projected_gd(C, f):
  x = ... # some initial solution
  xs = []
  for _ in range(T):
    x = x - lr * gradient(f) (x)
    x = project(C, x)
    xs.append(x)
  return mean(xs)
```
Here, the projection operator `x' = project(C, x)` is defined as solving the
minimization problem:
$ min_(x' in C) norm(x'-x). $
Note that this problem always have a global minimizer when $C$ is convex and
closed.

A common scenario is when $f$ is the average of the losses of individual data
points in the dataset:
$ f(w) = 1/n sum_(i = 1)^n cal(l)_i (w), $
where $cal(l)_i (w)$ is the loss w.r.t. the $i$-th data point in the dataset.

*Question*: How to make this algorithm private?

Similarly to $epsilon$-DP K-means, we simply add noise to the gradient:

Note that we have:
$ g = nabla f (w) = 1/n sum_(k = 1)^n cal(l)_i (w), $
then the $cal(l)_2$-sensitivity of $g$ is $(2 G)/n$, assuming that $norm(
  nabla
  cal(l)_i
) <= G$ at every data point. Hence, we simply add Gaussian noise $cal(N)(0,
  sigma^2 I)$, where $sigma^2 = (8 G^2 ln(2/delta))/(n^2 epsilon^2).$
```py
def private_gd(C, f):
  x = ... # some initial solution
  xs = []
  for _ in range(T):
    g = gradient(f) (x) + N(0, (8 G^2 ln(2/delta))/(n^2 epsilon^2) * I)
    x = x - lr * g
    x = project(C, x)
    xs.append(x)
  return mean(xs)
```

#theorem[
  Private GD is $(epsilon, delta)$-DP, for
  $sigma >= Omega((G sqrt(2 T ln(1/delta))/(n epsilon))).$
]

#proof[
  Use strong composition theorem, we have:
  $ sigma = Omega((G sqrt(2 T) ln(1/delta))/(n epsilon)), $
  By doing composition specifically for Gaussian, we can save a
  $sqrt(ln(1/delta))$ factor, which gives the desired result.
]

Now, since regular GD is inefficient, we mainly use SGD and BGD instead. In
general, if the batch size is $B$, then we simply replace the term $n$ by $B$.
However, we have this stronger result.

#lemma(title: "Amplification by sampling")[
  Consider an algorithm $A: X^* -> Y$. Let $S_(m, n): X^n -> X^m$ be the
  sampling algorithm (without replacement) $m$ of the orignal $n$ inputs.

  Then,
  if $A$ is $(epsilon, delta)$-DP, then $A compose S_(m, n)$ is $(epsilon',
    delta')$-DP, where
  $ epsilon' = ln(1 + (e^epsilon-1)m/n), delta' = delta m/n. $
]
Here, $epsilon' < epsilon$ and $delta' < delta$, a much better result than the
reduction we anticipated!

#proof[
  Let $x$ and $x'$ be neighboring datasets.
  Denote $S = S_(m, n), A' = A compose S_(m, n)$, and fix $S$.
  Then, let $I$ be the set of data point indices sampled by $S$, and $i$ be the
  index of the data point in which $x$ and $x'$ differs from each other, $y =
  S(x), y' = S(x')$, then:
  $
     PP[A'(x) in E] & = PP[A(S(x)) in E | i in I] PP[i in I] + PP[A(S(x)) in E | i
                        in.not I] PP[i in I] \
                    & = PP[A(y) in E | i in I] m/n + PP[A(y) in E | i
                        in.not I] (1 - m/n). \
    PP[A'(x') in E] & = PP[A(y') in E | i in I] m/n + PP[A(y') in E | i
                        in.not I] (1 - m/n). \
  $
  Denote $p = PP[A(y) in E | i in I], p' = PP[A(y') in E | i in I], q = PP[A(y)
    in E | i in.not I] = PP[A(y) in E | i in.not I]$, we have:
  $
    PP[A'(x) in E | i in I] <= exp(epsilon) min{PP[A'(x) in E | i
        in.not I], PP[A'(x') in E | i in I]} + delta.
  $
  which implies $p <= exp(epsilon) min{q, p'} + delta$. Similarly $p' <=
  exp(epsilon) min{q, p} + delta.$
  Finally, we need:
  $
    PP[A'(x) in E] = m/n p + (1-m/n) q <= (1 + m/n (exp epsilon - 1)) p' + delta
    m/n,
  $
  which can be verified easily.
]

If our total loss is $epsilon$, then each step can have a loss of $epsilon/n$
and the total number of steps is $approx n^2$ steps.

Finally, let's investigate the utility of projected/private GD.

#theorem[
  Given $f$ is convex, $G$-Lipschitz (i.e. $norm(nabla f) <= G$ everywhere), $C$ is
  a convex domain with diameter $R$, then define $x^*$ is the global minimizer
  of the minimization problem.

  Suppose $g$ in step $k$ satisfies:
  $ EE[g] = tilde(g), EE[norm(g - tilde(g))_2^2] = sigma^2, $
  where $tilde(g)$ is the value of `gradient(f)(x)` at that step.

  Define $G' = sqrt(G^2 + sigma^2)$, if the step size `lr` is $eta = R/(G'sqrt(T))$,
  then $f(hat(w)) - f(w^*) <= (R G')/sqrt(T).$
]

#proof[
  *Claim*: At step $t$, $EE[f(w)-f(w^*)] <= (eta norm(tilde(g))^2)/2 + 1/(2 eta) (
    norm(w - w^*)_2^2 - norm(w' - w^*)_2^2)$, where $w, w'$  denote the value of
  $w$ at this and the next step, respectively.

  We have:
  $ f(w) - f(w^*) <= ip(tilde(g), w - w^*) = 1/eta ip(eta tilde(g), w - w^*). $
  Note that:
  $
    2 ip(eta tilde(g), w - w^*) & = EE[2 ip(eta g, w-w^*)]  \
                                & = norm(eta g)_2^2 + norm(w-w^*)^2_2 -
                                  norm(eta g + w - w^*)_2^2 \
                                & = norm(eta g)_2^2 + norm(w-w^*)^2_2
                                  -EE[norm(w'-w^*)^2_2].
  $
  Then,
  $
    EE[f(w) - f(w^*)] <= 1/(2 eta) EE[norm(eta g)_2^2 + norm(w-w^*)^2_2
      -norm(w'-w^*)^2_2].
  $

  Now, since $hat(w)$ is the centroid of every $w$, we have:
  $
    EE[f(hat(w)) - f(w^*)] & <= 1/T sum_w EE[f(w) - f(w^*)]             \
                           & <= 1/T EE[sum_(g) (eta norm(g)_2^2)/2] + 1/(2 eta T)
                             (norm(w_0 -w^*)_2^2 - norm(w_T - w^*)_2^2) \
                           & <= (eta G'^2)/2 + (R^2)/(2 eta T).
  $
  Here, we have#footnote[$sum_k$ and $sum_g$ are taken over all steps of the
    algorithm.]:
  $
    EE[norm(g)_2^2] & = EE[norm(tilde(g) + (g - tilde(g)))_2^2] \
                    & = EE[norm(tilde(g))^2_2 + 2ip(tilde(g), g - tilde(g)) +
                        norm(g-tilde(g))_2^2]                   \
                    & = norm(tilde(g))_2^2 + 2ip(tilde(g), EE[g-tilde(g)]) +
                      sigma^2                                   \
                    & <= G^2 + sigma^2 = G'^2.
  $
  This bound is maximal when,
  $ S = (eta(G^2 +sigma^2))/2 = (R^2)/(2 eta T), $
  which implies $eta = R/(G'sqrt(T))$, $2S = R^2/(eta T) = (R G')/sqrt(T)$.
]

Plugging in stuff gives the following theorem:

#corollary[
  If $T = (epsilon^2 n^2) / d, eta = (R epsilon n)/(T G sqrt(d ln (1/delta)))$,
  then DP-PGD with $sigma = (2G sqrt(2T ln(1/delta)))/(n epsilon)$ has:
  $ EE[f(hat(w))-f(w^*)] <= cal(O) ((R G sqrt(d ln(1/delta)))/(epsilon n)). $
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

Given a dataset $D$ with $M$ samples. For different datasets $D$, we learn a
model $f(x)$, and let the expected prediction $macron(f) = EE_D [f]$.

Consider a noisy setting $y = underbrace(g(x), "real function") +
underbrace(epsilon, "noise")k($, then the expected error of an unseen sample $x$:
$
  & EE_(D, epsilon)[norm(f(x; D) - y)_2^2]             \
  & = EE_(D, epsilon)[norm(f(x; D)-g(x))_2^2 - underbrace(
        2 epsilon (f(x; D)-g(x)), #[0, since $EE[epsilon] = 0$]
      ) + epsilon^2]                                   \
  & = EE_D [norm(f(x; D)-g(x))_2^2] + sigma_epsilon^2,
$
where $sigma_epsilon$ is the error standard deviation (independent of $D$).

The first term is,
$
  & EE_(D)[norm(f(x; D)-g(x))_2^2]                                    \
  & = EE_(D)[norm(f(x; D)-macron(f)(x))_2^2 + norm(
        macron(f)(x) -
        g(x)
      )_2^2 + 2 (f(x; D)-macron(f)(x))(macron(f)(x) - g(x))]          \
  & = underbrace(EE_(D) [norm(f(x; D)-macron(f)(x))_2^2], "variance") +
    underbrace(norm(macron(f)(x) - g(x))_2^2, "bias squared") +
    2(macron(f)(x) - g(x)) underbrace(EE_D [f(x; D)-macron(f)(x)], 0)
$

This is the bias-variance decomposition.

== Feature maps

Consider ridge regression:
$ w = (X^T X + lambda I)^(-1) X^T y = X^T (X X^T + lambda I)^(-1) y. $
Then, if we let $G = X X^T, alpha = (G + lambda I)^(-1) y$,
$ w = X^T alpha => f(x) = w^T x = alpha^T X x = sum_i alpha_i ip(x_i, x), $
where $x_i$ are the data points in the training set.

Now, we use the *kernel trick*: to calc $X X^T$ and $ip(x_i, x)$, we only need to
calculate the inner products between input points, so if we don't want to define
a basis function for GLBM explicitly, we can skip that and define
$ K(x, x') = ip(phi(x), phi(x')). $

#theorem(title: "Mercer's theorem")[
  If $k$ is a SPD kernel:
  - $k(x, x') = k(x', x)$,
  - The Gram matrix calculated from $k$ is always PSD for all inputs,
  then there exists a feature space $HH$ and a feature map such that:
  $ k(x, x') = phi(x)^T phi(x). $
]

#example[
  Consider the RBF kernel in $RR$:
  $ k(x, x') = exp(-(x-x')^2/(2s^2)), $
  calculate the Taylor expansion yields
  $ phi(x) = exp(-x^2/(2s^2)) (1, sqrt(1/(s^2 1!) x), sqrt(1/(s^4 2!) x), ...), $
  which is infinite-dimensional.
]

#exercise[
  Given kernel $k$ with feature map $phi(dot)$ with infinite dimensions. How to
  find $z(dot)$ with finite dimensions that can approximate $k$.
]

#solution[
  Let $w tilde cal(N)_D (0, I)$ be a $D$-dimensional random vector. Consider the
  mapping:
  $ h(x) = exp(i w^T x), $
  then,
  $
    EE_w [h(x) h(y)^*] = EE_w [exp(i w^T (x - y))] = integral_(RR^D) p(w) exp(
      i
      w^T (x - y)
    ) dif w = exp(-1/2 norm(x-y)_2^2),
  $
  which is the Gaussian kernel.

  #theorem(title: "Bochner's theorem")[
    A continuous kernel $k(x, y)$ that is translational-invariant: $k(x, y) =
    k_s (x - y)$ is the Fourier transform of a non-negative measure.
  ]

  Then,
  $
    k(x, y) & = k_s (x - y) = integral p(w) exp(i w^T (x - y)) dif w \
            & = EE_w [exp(i w^T (x - y))]                            \
            & approx 1/R sum_(i = 1)^R exp(i w_i^T (x - y))          \
            & =
              underbrace(
                vec(
                  1/sqrt(R) exp(i w_1^T x),
                  1/sqrt(R) exp(i w_2^T x),
                  ...,
                  1/sqrt(R) exp(i w_n^T x)
                ), #[$h(x)$]
              )^T
              vec(
                1/sqrt(R) exp(-i w_1^T y),
                1/sqrt(R) exp(-i w_2^T y),
                ...,
                1/sqrt(R) exp(-i w_n^T y)
              )                                                      \
            & = h(x) h(y)^*
  $

  This is not the exact expression we want ($z(x)^T z(y) approx k(x, y)$), so we
  fine-tune by eliminating the imaginary component:
  $ z_w (x) = sqrt(2) cos(w^T x + b), w ~ W, b tilde cal(U)[0, tau] $
  Then,
  $
    EE[z_w (x) z_w (y)] = underbrace(EE_w [cos(w^T (x + y) + 2b)], #[0]) + EE_w
    [cos(w^T (x - y))]
  $
  Define $z$ similar to $h$:
  $
    z(x) = vec(
      1/sqrt(R) z_w_1 (x), 1/sqrt(R) z_w_2 (x), ..., 1/sqrt(R) z_w_n
      (x)
    ),
  $
  then $z(x)^T z(y) approx k(x, y)$.

  An alternative approach is:
  $ z_w_r (x) = vec(cos(w_r^T x), sin(w_r^T x)). $
]

== Kernel perspective of neural networks

When the width of a NN increases, the training loss improves, but the weight
changes less (relatively). So, we can approximate via Taylor's theorem:
$ f(x, w) approx f(x, w_0) + f'_w (x, w_0) (w - w_0). $
This is a linear (affine) function wrt $w$.

So MSE optimization of neural networks is simply linear regression. Though $f$
is not linear for $x$, it is for
$ phi(x) = f'_w (x, w_0) = nabla_w f (x, w_0)^T. $
This is called the neural tangent kernel.

// TODO: add the ODE thing here

== Principled DL

=== Transformers

The transformer takes an *sequence* input of size $N times D$, where each token
of the sequence is embedded as a $RR^(1 times D)$ vector. The output of one
layer is:
$ y = f("Attention"(x) + x), $

Where $"Attention"$ denotes the self-attention operation:
$
                   Q & = X W_Q^T                                                               \
                   K & = X W_K^T                                                               \
                   V & = X W_V^T                                                               \
  U = "Attention"(X) & = underbrace("softmax" ((Q K^T) / sqrt(D)), #[$A$: attention matrix])V.
$

Here, $U$ is the contextual representation of each token in the sequence.

*Problems:*
- Attention is ad-hoc: there is no theoretical result justifying the
  effectiveness of attention.
- Attention is not robust: corrupted training data massively hurts performance.
- Transformers suffer from oversmoothing: Token representations become identical
  as the network gets deeper.

*Solution:* We introduce a nonparametric kernel regression framework for
self-attention.

=== Ellipsoid Attention

We aim to estimate the distribution of input data points.

The most trivial way to do so is to partition the input space $X$ into finitely
many bins $X_1, X_2, ..., X_n$ and count:
$ p(x) = n_i/(n Delta_i) "if" x in X_i, $
where $n_i$ is the number of training data points in $X_i$.

KDE do this with:
$ p(x) = 1/Delta sum_(j = 1)^m K((x - x_i)/Delta), $
where $K$ is a kernel.

Now, returning to the setting of attention, consider each pair of key and value
vector of a token as a data point (with the key being the input, the value being
the output). Then, we want to calculate the value from the key in a noisy
setting:
$ v = f(k) + epsilon, $
so a natural choice for $f$ would be:
$ f(k) = EE[v|k] = integral_(RR^d) (v p(v, k))/(p(k)) dif v $
Substituting the KDE for $p$:
$
  f(k) = integral_(RR^d) (v sum_i phi_sigma (v-v_i) phi_sigma (k - k_i))/(sum_i
  phi_sigma (k - k_i)) dif v = (sum_i phi_sigma (k - k_i)integral_(RR^d) v
  phi_sigma (v-v_i) dif v)/(sum_i phi_sigma (k - k_i)).
$

We can recognize $ integral_(RR^d) v phi_sigma (v - v_i) dif v = v_i $
from the expectation of a normal distribution, so
$ f(k) = (sum_i v_i phi_sigma (k - k_i))/(sum_i phi_sigma (k - k_i)). $
Plug in $phi_sigma$:
$
  f(k)= (sum_i v_i exp (-norm(k - k_i)_2^2)/(2sigma^2))/(sum_i exp
  (-norm(k - k_i)_2^2)/(2sigma^2)) =
  (sum_i v_i exp (-(norm(k)_2^2 + norm(k_i)_2^2))/(2sigma^2) exp (k_i k^T)/(sigma^2))/
  (sum_i exp (-(norm(k)_2^2 + norm(k_i)_2^2))/(2sigma^2) exp (k_i k^T)/(sigma^2)).
$
If we force all $k_i$ to have the same length (via normalizing), this is reduced to:
$
  f(k) =
  (sum_i v_i exp (k_i k^T)/(2sigma^2))/
  (sum_i exp (k_i k^T)/(2sigma^2)) = sum_i "softmax" (exp (k^T k_i)/sigma^2) v_i.
$
This is the formula of self-attention if we plug in $k = q$.

== Linearized Self-attention

== Graph NN

Each graph is characterized by two matrices:
- $A in RR^(N times N)$: adjacency matrix
- $X in RR^(N times F)$: feature matrix

Update rule:
$
  h_i^((l + 1)) = sigma(h_i^((l)) W_0^((l))) + sum_(j in cal(N)_i) 1/c_(i j)
  h_j^((l)) W_1^((l)).
$

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
$ min D(T_\# cal(N), P) $

DM: add noises to data to turn it into Gaussian (fwd step), then denoise to turn
it into generated data.

$D$: KL-divergence, JS-divergence, *optimal transport*.

== Optimal transport formulations

=== General OT

We start with Monge's OT formulation, a problem of finding the minimum mapping
(in some sense) to transform from a probability distribution to another:

#definition(title: "Monge's OT")[
  Given two probability measures $mu, nu$ on spaces $X, Y$ and a cost function
  $c: X times Y -> RR$.

  Find the transform $T: X -> Y$ that minimizes:
  $ min_(T_\# mu = nu) integral_X c(x, T(x)) dif mu(x). $
]

Intutively: $T_\# mu$ is the pushforward of measure $mu$. Think of this as a
way to apply a function to a measure. A result that would shed some light on
this would be if $mu = delta_x$, then $nu = delta_(T(x))$.

We would focus on the case where $c = norm(dot)_2^2$ is the $cal(l)_2$-norm
squared.

*Problem*: if the input distribution $mu$ is too simple, then there would exist
no mapping $T$ such that $nu = T_\# mu$. Assuming $mu = delta_x$, then $nu$ can
only be something like $delta_y$, so everything would break when $nu = 1/2
delta_y_1 + 1/2 delta_y_2$.

#definition(title: "Kantorovich's OT")[
  Given two probability measures $mu, nu$ on spaces $X, Y$ and a cost function
  $c: X times Y -> RR$.

  Find the *transport plan* $gamma$, a joint probability measure on $X times Y$
  that minimizes:
  $ min_(gamma in Pi(mu, nu)) integral_(X times Y) c(x, y) dif gamma(x, y), $
  where $Pi(mu, nu)$ is the set of all probability measures on $X times Y$ with
  marginal distribution $mu$ and $nu$.
] <def:kanto-ot>

This is a relaxation of Monge's OT. Any transform $T$ in Monge's OT can be
turned into a Kantorovic's OT transport plan defined by:
$ gamma(x, y) = delta_y (T(x)). $

#example[
  If $mu = 1/n sum_(i = 0)^n delta_x_i$, $nu = 1/n sum_(i = 0)^n delta_y_i$,
  then Monge's OT and Kantorovich's OT are equivalent.
]

Note that $Pi(mu, nu)$ is always non-empty (aside from some trivial cases). And
if $gamma$ (in discrete case) is represented as a matrix, then the constraints
are linear constraints on rows and columns of this matrix.

=== Discrete OT formulation

Given $mu = sum_i mu_i delta_x_i, nu = sum_j nu_j delta_y_j$, then $gamma$ can be
represented as a matrix:
$ gamma_(i j) = gamma(x_i, y_j). $

Then, the marginal distributions of $gamma$ are:
$
  mu_i = mu({x_i}) & = gamma({x_i} times Y) = sum_j gamma(x_i, y_j)  \
  nu_i = nu({y_j}) & = gamma(X times {y_j}) = sum_i gamma(x_i, y_j).
$

#example[
  Solve for the case $n = m = 2$, $mu_1 = nu_1 = mu_2 = nu_2 = 1/2$. We have:
  $
    min c = gamma_(1 1) c_(1 1) + gamma_(1 2) c_(1 2) + gamma_(2 1) c_(2 1) +
    gamma_(2 2) c_(2 2)\
    "s.t." gamma_(1 1) + gamma_(1 2) = gamma_(2 1) + gamma_(2 2) = gamma_(1 1) +
    gamma_(2 1) = gamma_(1 2) + gamma_(2 2) = 1/2
  $

  Our matrix $gamma$ is in the form:
  $ gamma = 1/2 mat(t, 1 - t; 1 - t, t), $
  so $c = 1/2 (sum_(i j) c_(i j) + t(c_11+c_22-c_12-c_21)).$

  Clearly, either $t = 0$ (when $c_11+c_22>=c_12+c_21$) or $t = 1$ (otherwise).
]

#definition(title: "Kantorovich's OT LP formulation")[
  Given $mu in RR^n, nu in RR^m$. Then the LP formulation of @def:kanto-ot is:
  $
    min (mu, nu) = min_(gamma in RR^(n times m)) ip(C, gamma)\
    "s.t." gamma >= 0, gamma bold(1)_m = mu, gamma^T bold(1)_n = nu.
  $
  The feasible set is called the *transportation polytope*.
] <def:ot2>

#theorem[
  If $gamma$ is an extremal point of the transportation polytope. Then, $gamma$
  has at most $m + n - 1$ non-zero entries.
]

#proof[
  Consider a complete bipartite graph with vertex set $[n] union [m]'$.

  This is equivalent to the graph above, with edge set $E = {(i, j) gamma_(i j) >
    0}$. When so, the graph is a tree, which means $abs(E) <= abs([n]) +
  abs([m]') - 1 = n + m - 1$.
]

#theorem[
  The dual form of the problem in @def:ot2 has the following form:
  $
    max u^T p + v^T q,\
    "s.t." u bold(1)_m^T + bold(1)_n v^T <= c.
  $
]

The dual problem has only $n + m$ variables, much more simpler to solve than the
primal problem. Using the network simplex algorithm, this problem can be solved
in complexity:
$ cal(O)(max{m, n}^3, log max {m, n}) approx cal(O) (n^3) "when" m = n. $
There exists a theoretical optimal solution that is about $cal(O)(n^(2.5))$ (or
lower), but it is not practical.

=== Entropic Regularized OT

#definition(title: "Regularized OT")[
  The ROT of a transportation plan $gamma$ is defined as:
  $
    "ROT"_eta (mu, nu) = min_(gamma ...) underbrace(
      ip(C, gamma) + eta H(gamma),
      #[$g(gamma, eta)$]
    ),
  $
  where $H(gamma)$ is a regularization term.

  The entropic regularized OT (EROT) is the special case when $H(gamma) =
  -sum_(i, j) gamma_(i j) log gamma_(i j) = "KL"(gamma, mu times.circle nu)$,
  representing the entropy of $gamma$.
]

Clearly, since $H$ is strongly concave, the objective function in EROT is
strongly concave, which makes solving for EROT easier, as there is only exactly
one unique global minimizer.

Now, we consider a general result that holds for a wide range of $H(gamma)$.

#theorem[
  If the regularization term $H$ is non-negative and continuous,
  + ROT objective value converges to the OT objective value as $eta -> 0$.
  + ROT minimizer converges to the OT minimizer as $eta -> 0$.
]

#proof[
  Define $gamma(lambda)$ as the minimizer of ROT for $lambda$.
  + Given $lambda_1 > lambda_2 > 0$, then
    $
      g(gamma(lambda_1), lambda_1) >= g(gamma(lambda_1), lambda_2) >=
      g(gamma(lambda_2), lambda_2),
    $
    where the first inequality is due to $H >= 0$, and the second is due to the
    global minimum property of $lambda_2$.

    Hence, $g(gamma(lambda_k), lambda_k)$ is convergent for every monotonic
    decreasing $lambda_k$ that converges to $0$. Let the limit be $b$, and
    assuming that $b > g(gamma(0), 0)$, then there exists some $lambda$ such
    that $b >= g(gamma(lambda), lambda) > g(gamma(0), 0)$.

    However, $g(gamma(lambda), lambda) >= g(gamma(lambda), 0)$, which means
    $gamma(0)$ is not a minimizer of the original OT, a contradiction!
  + TODO
]

Though every $H$ is well-behaved, only one make the problem easier to solve, and
that is the entropy function.

Since $gamma_(i j) <= 0$ makes the objective function is $infinity$, we can
remove the $gamma >= 0$ constraint entirely.

Consider the dual problem:
$
  & max_(u in RR^n, v in RR^m) min_(gamma in RR^(n times m)) (ip(C, gamma) + eta
      H(gamma) + sum_(i = 1)^n u_i (sum_(j = 1)^m gamma_(i j) - mu_i) + sum_(j = 1)^m
      v_j (sum_(i = 1)^n gamma_(i j) - nu_i) )                    \
  & = max_(u in RR^n, v in RR^m) (min_(gamma in RR^(n times m)) (
        sum_(i = 1)^n sum_(j = 1)^m (u_i + v_j + C_(i j) - eta
          log(gamma_(i j))) gamma_(i j)) - ip(u, mu) - ip(v, nu))
$

Solving the inner problem:
$
  (partial cal(L))/(partial gamma_(i j)) = u_i + v_j + C_(i j) - eta (1 + log
    gamma_(i j)) = 0 => gamma_(i j) = exp ((eta e^(-1))/(u_i + v_j + C_(i j))).
$

Substituting back:
$
  & = max_(u in RR^n, v in RR^m) (min_(gamma in RR^(n times m)) (
        sum_(i = 1)^n sum_(j = 1)^m (u_i + v_j + C_(i j) - eta
          log(gamma_(i j))) gamma_(i j)) - ip(u, mu) - ip(v, nu)) \
  & = max_(u in RR^n, v in RR^m) underbrace(
      (-eta sum_(i, j) exp(
          (u_i + v_j -C_(i
          j))/lambda - 1
        ) + u^T mu + v^T nu), #[$f(u, v)$]
    ) + "constant".
$

This can be efficiently solved with Sinkhorn's algorithm (block coordinate
ascent):
+ Initialize $u, v$.
+ Fix $v$, update $u$ such that $(partial f)/(partial u) = 0$.
+ Fix $u$, update $v$ such that $(partial f)/(partial v) = 0$.
+ Return to step 2 until convergence.

#theorem(title: "Dvurechensky et al.")[
  Using Sinkhorn's algorithm, if the output of step $t$ is $gamma^t$, then:
  $ ip(gamma^t, C) <= ip(gamma^*, C) + epsilon, $
  when $t = cal(O) (1/ epsilon^2 norm(C)_infinity^2 log max{m, n})$
  and $eta = epsilon/(4 log max {m, n})$.
]

Each step has time complexity $cal(O)(max{m, n}^2)$, so the total time
complexity is
$
  cal(O) ((max{m, n}^2)/ epsilon norm(C)_infinity^2 log max{m, n}) approx
  cal(O)((max{m, n}^2)/epsilon^2),
$
"better" than the naive LP method. Here, we have a trade-off between dataset
size ($max{m, n}$) and the error ($cal(O)(log 1/epsilon)$ for LP and
$cal(O)(1/epsilon^2)$ for this approach).

== Deep Generative Models via OT

Popular choice for divergence in GANs is the Jensen-Shannon divergence:
$ "JS"(mu, nu) = 1/2("KL"(mu, (mu + nu)/2) + "KL"(nu, (mu + nu)/2)), $
where $"KL"$ denotes the Kullback-Leibler divergence:
$ "KL"(mu, nu) = integral_(RR^d) mu(x) log (mu(x)/nu(x)) dif lambda(x). $
This is problematic when $mu$ or $nu$ have:
- Disjoint supports.
- One is continuous while the other is discrete.

Note that $(mu+nu)/2$ denotes the mixture distribution between $mu$ and $nu$,
not the random variable that represents the average sample taken from $mu$ and
$nu$.

#example[
  If $mu = (phi, z)$ where $z tilde cal(U)[0, 1]$ and $nu = (0, cal(U)(0,
      1))$, then
  $ "JS"(mu, nu) = cases(log(2) "if" phi != 0, 0 "otherwise"). $

  We can verify this:
  - If $phi = 0$, then $mu = nu = (mu + nu)/2$, we have:
    $"JS"(mu, nu) = 2 "KL"(mu, mu) = 0$.
  - If $phi != 0$, then:
    $"JS"(mu, nu) = "KL"(mu, (mu + nu)/2) + "KL"(nu, (mu + nu)/2)$.
    Then, $mu(x, y) = delta(x-phi) bold(1)_[0, 1](y), nu(x, y) = delta(x)
    bold(1)_[0, 1](y)$.
    Calculating the first term:
    $
      "KL"(mu, (mu + nu)/2) & = integral_(RR^2) mu(x, y) log (mu(x, y))/((mu+nu)/2(x, y))
                              dif lambda_2 (x, y)                               \
                            & = integral_(RR) delta(x - phi) log
                              (2delta(x-phi))/(delta(x-phi)+delta(x)) dif mu(x) \
                            & = log((2 delta(0))/(delta(0)+ delta(phi)))        \
                            & = log 2.
    $
  Similarly, $"KL"(nu, (mu + nu)/2) = log 2.$ Hence, $"JS"(mu, nu) = log 2$.
]

Wasserstein metric solves this problem. In the above example, the Wasserstein
metric is always $abs(phi)$ for every $phi in RR$.

#definition(title: "Wasserstein metric")[
  The Wasserstein metric between $mu$ and $nu$ is defined as:
  $
    W_1 (mu, nu) = inf_(gamma in Gamma(mu, nu)) integral norm(x - y) dif
    gamma(x, y).
  $
]

#theorem(title: "Dual Wasserstein metric")[
  The Wasserstein metric between $mu$ and $nu$ can be calculated as:
  $
    W_1 (mu, nu) = sup_(f in cal(L)^1) (EE_(x tilde mu) [f(x)] - EE_(y tilde nu)
      [f(y)]),
  $
  where $cal(L)^1$ denotes the set of 1-Lipschitz functions: $abs(f(x)-f(y)) <=
  norm(x-y), forall x, y in RR^d$.
]

= Mixture of Experts

Idea: combine multiple simple learners (experts) to form a complex learner.

#definition[
  Given gating functions $pi_1, pi_2, ..., pi_n$, where $pi_k: RR^d -> [0, 1]$.
  A mixture of experts (MoE) system with experts $f_1, ..., f_M$, which
  $f_k: RR^d -> RR^(d')$ is the function $f: RR^d -> RR^(d')$ defined by:
  $ f(x) = sum_(k=1)^M pi_k (x) f_k (x). $
]

#example[
  A simple set of gating functions is the softmax gating functions:
  $
    pi_i^"softmax" (x) prop exp(beta_(1 i)^T x + beta_(0 i)) "such that"
    pi_i^"softmax" "sums to" 1.
  $
  Another choice for gating functions is the sigmoid gating functions:
  $ pi_i^"sigmoid" (x) = 1/(1 + exp(-(beta_(1 i)^T x + beta_(0 i)))). $
  and its normalized variants:
  $ pi_i^"n_sigmoid" (x) prop pi_i^"sigmoid" "such that" pi_i^"n_sigmoid" "sums to" 1. $
  A set of gating functions useful for domain generalization is the cosine
  gating functions:
  $
    pi_i^cos (x) = pi_i^"softmax" ((beta_(1 i)^T x)/(norm(beta_(1 i))norm(x)) +
      beta_(0 i)).
  $
  For medical AI (and multimodal applications),
  a typical choice is the Laplace gating functions:
  $ pi_i^"laplace" (x) = pi_i^"softmax" (-norm(b_(1 i) - x) + b_(0 i)). $
]

The goal of MoE is to scale massive AI models without sacrificing computational
cost. Popular approaches include:
- Sparse MoE: only train one expert for each input (the expert with the highest
  gating weight).
- Switch Transformer:
- ...

In learning theory, we are interested in these fundamental questions regarding
MoE:
- How good can MoE approximate the ground truth?
- How well can experts be learned from MoE?

#theorem(title: "Assaf et al., 1998")[
  We have:
  $ sup_(f' in cal(L)_p^s) inf_(f in "MoE"_"const") norm(f - f')_2 <= c/M^(s/d), $
  for some universal constant $c$,
  $norm(dot)_2$ is taken in a bounded set $I$, and

  $cal(L)_p^s$ denotes the Sobolev space (functions with all $k$-th order
  derivatives of $f$ is bounded, provided that $1 <= k <= s$),

  $"MoE"_"const"$ is the set of MoE systems generated by constant experts with softmax
  gates.
]

As $M -> infinity$, MoE systems can approximate any complex function.

*Question*: What about other gates: sigmoid gates, cosine gates, Laplace gates?

So we know MoE systems can approximate nice functions. Now, we will be focusing
on training MoE systems. Formally, given data points $(x_k, y_k)_(k in [n])$,
such that $y_k tilde cal(N)(f^* (x_k), Sigma).$ We want to learn a MoE $f$
that approximates $f^*$ well.

Denote $f_(beta, eta) (x) = sum_(k=1)^M pi_k (x|beta_k) f_k (x|eta_k).$

Maximum likelihood estimation: we solve for
$ min_(beta, eta) 1/n sum_(k = 1)^n norm(y_k - f_(beta, eta) (x_k))_2^2. $

In the easy case where $f^*$ can be represented as a MoE system
with the same number of experts as $f$,
$ f^* = f_(beta^*, eta^*), $
then we have the following result:
$ norm(f_(hat(b), hat(nu)) - f)_2 <= c sqrt(log(n)/n). $

When $pi$ is softmax, then
$
  sup_(x) norm(f_(beta, eta) (x) - f_(beta^*, eta^*) (x)) >= c sum_(i=1)^M
  sum_(j in V_i) (norm(n_j - n_i^*) + norm(beta_(1j) - beta_(1i)^*) +
  abs(exp(beta_(0j) - exp(beta^*_(0i)))).
$
Here, $V_i = {eta: norm(eta - eta^*_i) = min_(j in [M]) norm(eta - eta^*_j) }$.

== Finetuning and MoE systems

Foundation models contain a lot of knowledge, but they are not trained for
specific tasks. So, we need to finetune them to adapt to the specific tasks.
Finetuning is a training process that is much less expesive than training from
scratch.

An attention layer can be described as:
#let softmax = math.op("softmax")
$ T = f_cal(l) ("Attention"(X) + X) = f_cal(l) (softmax((Q K^T) / sqrt(D)) V + X), $
where $Q, K, V$ are the query, key, and value matrices, defined as:
$ Q = X W_Q^T, K = X W_K^T, V = X W_V^T, $
where $W_Q, W_K, W_V$ are the weight matrices of the attention layer.

Finetuning via prefix tuning can be described as: we append a prompt into the
input in $RR^(N times d)$ to get a $RR^((N + L) times d)$ matrix, then we pass
that into the transformer. Then, we freeze the transformer and optimize the
prefix prompt.

Then, there is a funny result:
#theorem[
  Each row of self-attention can be represented as a MoE system.
]

#proof[
  We adopt the following notation from matrix algebra: $A^i$ as the $i$-th row
  of $A$, and $A_i$ as the $i$-th column of $A$. For the motivation and
  properties of writing it down this way, read my funny intro to optimization
  book.

  We have:
  $ Q K^T = X W_Q^T W_K X^T $
  Then, each row of $A = softmax((Q K^T)/sqrt(D)) V$ is:
  $
    A^i = underbrace(softmax((Q K^T)/sqrt(D))^i, "gating") underbrace(
      X W_V^T,
      "expert"
    )
  $
  Since softmax is row-wise,
  $ softmax((Q K^T)/sqrt(D))^i = softmax((Q K^T)^i/sqrt(D)) = softmax((Q^i K^T)/sqrt(D)), $
  which is
  $ softmax((X^i W_Q^T W_K X^T)/sqrt(D)). $
  Then, we can see that $A^i$ is equivalent to the MoE with:
  $
    pi (X) & = softmax((X^i W_Q^T W_K X^T)/sqrt(D)) && (in RR^(1 times N)), \
      f(X) & = X W_V^T                              && (in RR^(N times D)).
  $
]

== LoRA finetuning

Since $W_K$, $W_Q$ and $W_V$ are very large matrices, we will finetune those
using:
$
  W_K += B_K A_K, B_K in RR^(m times r), A_K in RR^(r times d),\
  W_Q += B_Q A_Q, B_Q in RR^(m times r), A_Q in RR^(r times d),\
  W_V += B_V A_V, B_V in RR^(d times r), A_V in RR^(r times d).
$

Notable derivatives:
- DoRA (NVIDIA): $W_K := (W_K B_V A_V)/norm(W_K B_V A_V),$ and similarly...
- VeRA (Qualcomm): $W_K += B_Q Gamma_(B Q) A_Q Gamma_(A Q)$, where $Gamma_(B
  Q)$ and $Gamma_(A Q)$ are learnable diagonal matrices.

= Spectral Graph Theory

Main idea: Represent graphs as matrices, then study the eigenvalues and
eigenvectors of these matrices.

*Notation*: For a given index tuple $I = (i_1, ..., i_n)$, for a matrix $A in
RR^(r times c)$ denote
$
  A_I & = mat(A_i_1, ..., A_i_n)     & in RR^(r times n) & "as" A "with columns" i_1, ...,
                                                           i_n, \
  A^I & = mat(A^(i_1); ...; A^(i_n)) & in RR^(n times c) & "as" A "with rows" i_1, ...,
                                                           i_n.
$

When $I = (i)$ contains only one index, we can write it as $A_i$ or $A^i$.

Then, we have this elegant result.
#theorem[
  For any two matrices $A, B$ such that the product $A B$ makes sense, we have:
  $ (A B)^I_J = A^I B_J. $
]

If we write the row index before the matrix, then we have:
$ attach((A B), tl: I, br: J) = attach(A, tl: I) B_J, $
which kinds of makes more sense, but it looks somewhat unnatural, so we won't
use them here.

The standard basis of $RR^n$ is denoted as $delta_i$, for $i in [n]$.

== Graphs

#definition(title: "(Symmetric) graph")[
  A graph $G$ is a pair $(V, E)$, where $V$ is the set of vertices and $E
  subset.eq {{u, v}: u, v in V}$ is the set of edges.
]

For convenience, we define getters: $V(G)$ as the vertex set of $G$,  $E(G)$ as
the edge set of $G$.

Some notations:
- $abs(G) = abs(V(G))$ as the number of vertices.
- $e(G) = abs(E(G))$ as the number of edges.
- Two vertices $u, v$ are *adjacent* if ${u, v} in E(G)$.
- The *degree* of a vertex $v$ is the number of edges incident to $v$:
  $deg_G (v) = abs({v in V(G): {u, v} in E(G)})$. This can also be denoted as
  $deg(v)$ if $G$ is not ambigious.

Then, we can represent a graph as an adjacency matrix $a$: given a bijection $I$
from $V(G)$ to $[abs(G)]$ (an indexing of the vertices), we have:
$
  A = A(G) = (a_(I(u), I(v)))_(u, v in V(G)), "where" a_(I(u), I(v)) = chi_E(G)
  ({u, v}).
$

For convenience, we will treat graph vertices as integers in $[abs(G)]$.

#let diag = math.op("diag")
#definition(title: "Laplacian matrix")[
  The Laplacian matrix of a graph $G$ is defined as:
  $ L(G) = D(G) - A(G), $
  where $D(G) = diag(d)$, $d_(u) = deg(u)$.

  In other words,
  $
    L(G)_u^v = L(G)^v_u = cases(
      deg(u) "if" u = v,
      -1 "if" u != v "and" {u, v} in E,
      0 "otherwise"
    ).
  $
]

We have the following trivial results.
#theorem[
  - Entries within a column/row of $L(G)$ sums to 0.
  - If $L_e$ denotes the Laplacian matrix of a graph $G$ with only edge $e$, then
    $L(G) = sum_(e in E) L_e$.
]

Now, consider some vector $x in RR^abs(G)$. We have:
$
  (L x)^(u) = sum_(e in E) L_e^u x = sum_(e in E, e = {u, v}) (x^u - x^v).
$

== Eigenvalues and eigenvectors of Laplacian matrices

For a vector $x in RR^abs(G)$, we have
$ L_{u, v} x = (x^u - x^v) delta_u + (x^v - x^u) delta_v, $
which implies
$
  x^T L_{u, v} x & = sum_(i) (x^T)_i ((x^u - x^v) delta_u + (x^v - x^u)
                     delta_v)^i                    \
                 & = sum_i ((x^i) (x^u - x^v) delta^i_(u) + (x^i) (x^v - x^u)
                     delta^j_(u))                  \
                 & = x^u (x^u-x^v) + x^v (x^v-x^u) \
                 & = (x^u - x^v)^2.
$
By linearity, we have:
$ x^T L x = x^T sum_({u, v} in E) x = sum_({u, v} in E(G)) (x^u-x^v)^2 >= 0. $

Since $L$ is a real symmetric matrix, all its eigenvalues are real. Moreover, there
is an orthonormal basis of $RR^abs(G)$ consisting of eigenvectors of $L$.

Denote the eigenvalues of $L$ as $lambda_1, lambda_2, ..., lambda_n$ and the
corresponding basis eigenvectors be $v_1, v_2, ..., v_n != 0$, where $n =
abs(G)$.

Then,
$ v_i^T L v_i = lambda_i norm(v_i)_2^2 >= 0, $
which implies $lambda_i >= 0$. Hence, all eigenvalues of $L$ are non-negative.

Moreover, from
$ L x = sum_(e in E, e = {u, v}) (x^u - x^v), $
we know that if $x^u = vec(1, ..., 1)$, $L x = 0$, so the smallest eigenvalue of
$L$ is 0.

#theorem[
  The multiplicity of the eigenvalue 0 of $L$ is equal to the number of
  connected components of $G$.
]

#proof[
  If $G$ can be split to $m$ connected components, then we can write $L$ as a
  diagonal-block matrix like so:
  $ L = mat(L_1, 0, ..., 0; 0, L_2, ..., 0; 0, 0, ..., L_m), $
  so our argument above applies for $x$ in the form of:
  $ x^k = sum_(i in V(G_k)) bold(e)_i, $
  where $G_k$ denotes the $k$-th connected component of $G$.
  There are $m$ such vectors, which are linearly independent, so the
  multiplicity of 0 is at least $m$.

  Now, if $x$ is an eigenvector of $L$ with eigenvalue 0, then
  $ x^T L x = sum_({u, v} in E) (x^u - x^v)^2 = 0. $
  This implies $x^u = x^v$ if ${u, v} in E$, and more generally when $u$ and
  $v$ are connected. Hence, $x$ must be a linear combination of the vectors
  $x^k$ defined above. Hence, $x^k$ is the basis of the eigenspace of eigenvalue
  0.
]

#theorem[
  If $0 = lambda_1 <= lambda_2 <= ... <= lambda_n$ are the eigenvalues of $L$,
  then each $lambda_k$ can be recursively defined via
  $ lambda_k = min {x^T L x: x in V_(k - 1)^perp, norm(x)_2 = 1 }, $
  where $V_(k - 1)$ is the basis formed by $b_1, b_2, ..., b_(k-1)$, the first
  $k-1$ eigenvectors of $L$.
] <thr:variational>

#proof[
  For each $x in V_(k - 1)$, we can represent $x = sum_(i >= k) x^i b_i$, then:
  $
    x^T L x & = (sum_(i>=k) x^i b_i)^T (sum_(j>=k) x^j lambda_j b_j)             \
            & = sum_(i, j >= k) x^i x^j lambda_j b_i^T b_j                       \
            & = sum_(i>=k) lambda_i (x^i)^2                                      \
            & = sum_(i>=k) (lambda_i - lambda_k) (x^i)^2 + lambda_k norm(x)_2^2,
  $

  Clearly, if $norm(x)_2 = 1$, then this sum is minimized when:
  $ sum_(i>=k) (lambda_i - lambda_k) (x^i)^2 = 0, $
  which can happen when $x^i = delta^i_(k)$. Then, and $x^T L x = lambda_k$.

]

== Spectral clustering

Consider a graph $G = (V, E)$ with $abs(G) = n$. Our aim is to find *clusters*
within $G$.

#let vol = math.op("vol")
#definition[
  Let $G = (V, E)$ be a graph.

  The *volume* of a set $S subset.eq V$ is the sum of all degrees of vertices in
  $S$: $ vol S = sum_(v in S) deg(v). $

  The *conductance* of a set $S subset.eq V$ is defined as:
  $
    phi.alt_G (S) = abs(E(S, V without S))/min{vol S, vol (V without S)},
  $
  where $E(X, Y) = {{x, y} in E: x in X, y in Y}$.
]

The idea is that clusters are subsets $S subset.eq V$ with low conductance.

Define
$ phi.alt_G = min_(diameter != S subset.neq V) phi.alt_G (S). $

If $G$ is connected, then $phi.alt_G > 0$. Otherwise, we can take $S$ to be
the set of vertices in one connected component, then $phi.alt_G = 0$.

Let's consider the case where $G$ is $d$-regular: $deg v = d, forall v in V$.
Then, consider the following:
$ cal(L) = 1/d L = I - 1/d A, $

Then,
$
  phi_G (S) = abs(E(S, V without S))/min{vol S, vol (V without S)} = abs(
    E(S, V
      without S)
  )/(d min{abs(S), abs(V without S)})\
  => phi_G = min_(diameter != S subset.neq V) abs(E(S, V without S))/(d
  min{abs(S), abs(V without S) }) = min_(diameter != S subset.neq V, 0 < abs(S)
  <= n/2) abs(E(S, V without S))/(d abs(S)).
$

#lemma[
  Every eigenvalues $lambda_i$ of $L(G)$ is in $[0, 2]$.
]

#proof[
  See Gershgorin circle theorem.
]

#lemma(title: "Discrete Cheeger Inequality")[
  $lambda_2/2 <= phi_G <= sqrt(2 lambda_2)$.
]

#proof[
  *Lower bound*:
  Note that $lambda_2$ (of $L$) is:
  $ lambda_2 = min{x^T L x: x perp x_1, norm(x) = 1}, $
  where $x_1$ is the first eigenvector of $L$, which is the constant vector
  $mat(1, ..., 1)^T.$
  This can be rewitten as:
  $ lambda_2 = min {(x^T cal(L) x)/norm(x)_2^2, x perp mat(1, ..., 1)^T}. $

  Letting $x^u = chi_S (u) + t$, then:
  $
    x^T cal(L) x = 1/d sum_({u, v} in E) (x^u - x^v)^2 = 1/d abs(
      E(S, V without
        S)
    ),
  $
  where $t$ is a normalizing constant to make $x perp b_1 = mat(1, ..., 1)^T$.
  Now, we find $t$:
  $
    b_1^T x = sum_i x^i = sum_i (chi_S (i) + t)
    = abs(S) + n t => t = -abs(S)/n.
  $

  Then, we have:
  $
    norm(x)_2^2 & = sum_i (x^i)^2                                 \
                & = sum_(u in S) (1 + t)^2 + sum_(u in.not S) t^2 \
                & = n t^2 + 2t abs(S) + abs(S)                    \
                & = abs(S)^2/n - 2 abs(S)^2/n + abs(S)            \
                & = abs(S) - abs(S)^2/n.
  $
  Considering subsets $S subset.eq V$ with $abs(S) <= n/2$, we have:
  $ norm(x)^2_2 >= abs(S) - 1/2 abs(S) = abs(S)/2. $
  Hence,
  $ (x^T cal(L) x)/norm(x)_2^2 <= 2 dot abs(E(S, V without S))/(d abs(S)), $
  so
  $
    lambda_2 = min{(x^T cal(L) x)/norm(x)_2^2: x perp b_1 } <= 2 min_(
    S subset.neq V, 0 < abs(S) <= n/2) dot abs(E(S, V without S))/(d abs(S)) =
    2 phi.alt_G.
  $
  Hence, $phi.alt_G >= lambda_2 / 2$.

  *Upper bound*: We aim to construct $S$ with $phi_G (S) <= sqrt(2 lambda_2)$
  via an algorithm:
  - Fix $x in RR^abs(V),$ sort the vertices such that $x_v_1 <= ... <= x_v_n$.
    This gives us $n - 1$ different cuts $({v_1}, {v_2, ..., v_n}), ({v_1, v_2},
      {v_3, ..., v_n}), ..., ({v_1, ..., v_(n-1)}, {v_n})$.
  - Output the cut with the smallest conductance.

  #lemma[
    If $y perp b_1$, then the algorithm above output a vertex set $S$ with at
    most:
    $ phi_G (S) <= (sum_({u, v} in E) abs(y^u - y^v))/(d sum_u y^u), $
  ]

  #proof[

  ]

  Now, the only work remaining is to pick some $y$ such that:
  $
    R_G (x) = (x^T cal(L) x)/norm(x)_2^2 = 1/(d norm(x)_2^2) sum_({u, v}
    in E(G)) (x^u-x^v)^2
  $
  satisfies
  $ sqrt(2 R_G (x)) >= (sum_({u, v} in E) abs(y^u - y^v))/(d sum_u y^u) $
  and $y perp b_1$.

  This is equivalent to:
  $
    (sum_({u, v} in E) (x^u - x^v)^2)/(d norm(x)_2^2) >= ((sum_({u, v} in E)
      abs(y^u - y^v))/(d sum_u y^u))^2,
  $
  which holds when $y^u = (x^u)^2$ by the Cauchy-Schwarz inequality.

  We have $lambda_2 = min {R_G (x): x perp x_1}$. Then, we simply pick $x
  perp x_1$ such that $lambda_2 = R_G (x)$. By the lemma, we have:
  $ phi.alt_G (S) <= sqrt(2 R_G (x)) = sqrt(2 lambda_2). $
]

== Spectral sparsification

Since graph matrices grows quadratically with the number of vertices, we want to
sparsify the graph, i.e., reduce the number of edges while preserving the
spectral properties of the graph.

For every $S subset.eq V$, denote $bold(1)_S$ as the vector with $bold(1)_S^u =
chi_S (u)$. Recall that $$.

Here are some motivations for spectral sparsification:
- *Cut sparsifiers*:
  Given a graph $G = (V, E)$, we want to find a sparse $H = (V, E')$ such that
  $ abs(E_G (S, V without S)) approx abs(E_H (S, V without S)), $
  for every $S subset.eq V$.

  *Benczur-Karger*: gives a multiplicative approximation of the cut sparsifier:
  $ abs(E_G (S, V without S)) approx abs(E_H (S, V without S)) dot (1 + epsilon), $
  where $epsilon$ is a small constant with $abs(E_H) = O((n log n)/epsilon^2)$.

  This is equivalent to $(R_G (bold(1)_S))/(R_H (bold(1)_S)) <= 1 + epsilon$.
- *Learning on graphs*: We have a graph $G = (V, E)$, some vertices in $V$ are
  labeled, and we want to learn a function $f: V -> RR$ that minimizes,
  $ sum_({a, b} in E) w_(a, b) (f(a)-f(b))^2, $
  subjected to the labeled values of $G$.

  Given a cut $S subset.eq V$, recall that $R_G (bold(1)_S) = abs(
    E_G (S, V
      without S)
  ),$ with $bold(1)_S^u = cases(1 "if" u in S, 0 "otherwise")$.
  This result can be generalized to the *weighted* Laplacian matrix $cal(L)_G^w$,
  which gives the form of the above minimization problem as $bold(1)_S^T cal(L)_G^w
  bold(1)_S$.

  Then, *spectral sparsification* (Spielman-Teng) gives us a sparse graph $H$
  with:
  $ (x^T cal(L)_H x)/(x^T cal(L)_G x) in [1/(1+epsilon), 1+epsilon], forall x in
  RR^abs(G), $ (division by zero follows $1/0=infinity, 0/0=0$),
  which can be used to solved the above problem with a similar
  guarantees as the original graph $G$. We call $H$ as an
  $epsilon$-approximation of $G$

Do note that, spectral sparsification is not the same as cut sparsification, but
rather a generalization of it. In fact, cut sparsification is a special case of
spectral sparsification, where the inequality only has to hold for binary
vectors $bold(1)_S$ for every $S subset.eq V$.

*Notation*:
- Given weighted graph $G = (V, E, w)$, denote $k G$ as the same graph
  as $G$, but with weights multiplied by $k$.
- Given weighted graphs $G, H$, we write $G lt.tilde H$ if $cal(L)_H -
  cal(L)_G$ is positive semidefinite, or equivalently:
  $ x^T cal(L)_H x >= x^T cal(L)_G x. $

#theorem[
  $H$ is an $epsilon$-approximation of $G$ if and only if
  $ 1/(1 + epsilon) G lt.tilde H lt.tilde (1 + epsilon) G, $
  or equivalently:
  $
    cal(L)_H - cal(L)_(G) lt.tilde epsilon cal(L)_G "and" cal(L)_G - cal(L)_H
    lt.tilde epsilon cal(L)_H,
  $
  where $lt.tilde$ denotes the Loewner order (basically $A gt.tilde 0$ if $A$ is
  positive semidefinite).
]

Implications of spectral sparsification:
- Similar boundaries $abs(E (S, S'))$.
- $cal(L)_H$ and $cal(L)_G$ has similar eigenvalues: $lambda_i (L_H) approx
  lambda_i (cal(L)_G)$ (by @thr:variational).
- Solutions of linear equations w.r.t. the Laplacian matrix are similar:
  $ cal(L)_G x = b, cal(L)_H y = b => x approx y. $
  More precisely, $norm(x - y)_cal(L)_G <= epsilon norm(x)_cal(L)_G$, where
  $norm(dot)_cal(L)_G$ is defined as:
  $ norm(x)_cal(L)_G = sqrt(R_G (x)). $
- Solutions of the graph learning problems are similar.

#proof[
  We will prove the third statement. Since $cal(L)_G x = cal(L)_G y$, we have:
  $ (cal(L)_H - cal(L)_G) y = cal(L)_G x - cal(L)_G y = cal(L)_G (x - y). $
  Then,
  $ (x-y)^T cal(L)_G (x - y) <= $
]

Now, we will give a method to construct spectral sparsifiers via *random
sampling*.

If $H$ is a random graph such that $EE [cal(L)_H] = cal(L)_G$, then matrix
concentation gives us that $H$ is an $epsilon$-approximation of $G$ with high
probability $1 - alpha$.

Then, we can construct $H$ as follows:
- Assign a probability $p_e$ to each edge $e in E$ (we'll do this later)
- Pick edges in $E$ with the probability assigned above. If included, then
  set the weight to be $w_e/p_e$.

We have:
$ cal(L)_G = sum_(e in E) w_e L_e = sum_({a, b} in E) w_{a, b} v_(a, b) v_(a, b)^T, $
where $v_(a, b) = delta_a - delta_b$.
Then,
$
  cal(L)_H = sum_({a, b} in E(H)) w_{a, b}/p_{a, b} (v_(a, b) v_(a, b)^T) => EE[cal(L)_H] = sum_(e in
  E) p_{a, b} dot w_{a, b}/p_{a, b} (v_(a, b) v_(a, b)^T) = cal(L)_G.
$

Now, we let $p_e = 1/r w_e v^T_(a, b) cal(L)_G^(-1) v_(a, b)$ where $e = {a, b}$,
here, $cal(L)_G^(-1)$ denotes the pseudoinverse of $cal(L)_G$.

We have the following lemma

#lemma[
  If $G$ is connected, then
  $ sum w_e v^T_(a, b) cal(L)_G^(-1) v_(a, b) = n - 1. $
]

#proof[
  For simplicity, assuming $w_(a, b) = 1$, then we have:
  $ v_(a, b)^T cal(L)_G^(-1) v_(a, b) = PP[{a, b} in T(G)], $
  where $T(G)$ is an uniform spanning tree of $G$.
  Then,
  $
    sum_({a, b} in E) v_(a, b)^T cal(L)_G^(-1) v_(a, b) = sum_({a, b} in E) PP[{a,
        b} in T(G)] = n - 1,
  $
  hence the desired result.

  For the general case, we have:
  $
    sum_({a, b} in E) w_(a, b) v_(a, b)^T cal(L)_G^(-1) v_(a, b)
    &= sum_({a, b} in E) w_(a, b) tr(v_(a, b)^T cal(L)_G^(-1) v_(a, b))\
    &= sum_({a, b} in E) w_(a, b) tr(cal(L)_G^(-1) v_(a, b) v_(a, b)^T)\
    &= tr(
      cal(L)_G^(-1) underbrace(
        sum_({a, b} in E) w_(a, b) v_(a, b) v_(a,
        b)^T, cal(L)_G
      )
    )\
    &= tr(cal(L)^(-1)_G cal(L)_G) = n.
  $
  But because pseudoinverse bs, it is actually $n - 1$, not $n$.
]

We want $p_e in [0, 1], forall e in E$ and $EE[e(H)]$ to be small. Calculating
the latter:
$ EE[e(H)] = 1/r sum_(e in E) PP[e in E(H)] = 1/r sum_(e in E) p_e = (n - 1)/r. $

We set $r = (c epsilon^2)/(log n)$. Then, apply the following lemma,

#lemma(title: "Chernoff's inequality")[
  Let $X_1, ..., X_n$ be independent random variables with $0 <= X_i <= 1$,
  and $X = sum_(k = 1)^n X_k, mu = EE[X]$. Then,
  $ PP[abs(X-mu) >=delta mu] <= 2 exp(-(delta^2 mu)/3), forall delta in (0, 1/3). $
]

we have $EE[e(H)]$ is concentrated around its mean, i.e. $(n-1)/r = cal(O)((n
  log n)/epsilon^2)$.

Finally, we need that: with high probability, $cal(L)_H$ is an
$epsilon$-approximation of $cal(L)_G$. To prove this, we need the matrix
concentration inequality:

#lemma(title: "Matrix concentration inequality")[
  Let $X_1, ..., X_n$ be positive semi-definite matrices, and $X = sum_(k=1)^n
  X_k$ so that $norm(X_k) <= M$ almost surely. Then,
  if $lambda_min, lambda_max$ are the min and max eigenvalues of $X$,
  $mu_min, mu_max$ are the min and max eigenvalues of $EE[X]$,
  $
    PP[lambda_min (X) <= (1 - epsilon) mu_min] <= n
    (e^(-epsilon)/(1-epsilon)^(1-epsilon))^(mu_min / M),\
    PP[lambda_max (X) <= (1 + epsilon) mu_max] <= n
    (e^(epsilon)/(1+epsilon)^(1+epsilon))^(mu_max / M).
  $
]

WLOG assuming $EE[X] = I$, so $mu_min = mu_max = 1$. Now, for every PD matrices
$A, B$, such that:
$ A lt.tilde (1 + epsilon) B => B^(-1/2) A B^(-1/2) lt.tilde (1 + epsilon) I. $
This applies to our Laplacian matrices, so we want to prove:
$
  1/(1+epsilon) <= lambda_min (cal(L)_G^(-1/2)cal(L)_H cal(L)_G^(-1/2))<= lambda_max (underbrace(cal(L)_G^(-1/2) cal(L)_(H) cal(L)_G^(-1/2), Pi)) <= 1 +
  epsilon.
$

Clearly, since $EE[cal(L)_H] = EE[cal(L)_G]$, the matrix $Pi$ has mean $I$.
Then, we can apply the lemma with each $X_e = cal(L)_G^(-1/2) w_e/p_e eta_e
v_(a, b) v_(a, b)^T cal(L)_G^(-1/2)$ for each $e = {a, b} in E(G)$, where
$eta_e$ is a Bernoulli random variable with $p = p_e$.

Then, the probability of $PP[lambda_max (Pi) >= 1 + epsilon]$ is at most
$ n exp(-epsilon^2/(3 r)) = n^(-1), $
given $c <= 1/6.$

Similarly, we can derive the probability of $PP[lambda_min (Pi) <= 1/(1 +
  epsilon)]$ is at most $n^(-1)$. This can be combined to give the desired
result.

== Random walks and local clustering algorithms

=== Random walks (RWs)

Let $G = (V, E, w)$ be a weighted graph. A random walk (RW) on $G$ is a
stochastic process that starts at a vertex $v_0 in V$ and at each step, moves to
a neighbor vertex $v_1 in V$ with probability proportional to the weight of
the edge connecting $v_0$ and $v_1$.

Now, denote $p_t$ as the probability distribution of the RW at time $t$, i.e.,
$p(t)^v = PP["at" v "at time" t].$ We fix the starting node to make this
well-defined.

Then, we have:
$
  p(t)^v = sum_({u, v} in E) p(t)^u w_{u, v}/(d^w (u)) = sum_(u in V)
  p(t)^u w_{u, v}/(d^w (u)),
$
where $d^w$ denotes the weighted degree of $v$:
$ d^w (u) = sum_({u, v} in E) w_{u, v}, $
and if ${u, v} in.not E$, we set $w_{u, v} = 0$.

Writing this in matrix form (consider each $p_t$ as a vector):
$ p^v_(t + 1) = sum_(u in V) underbrace(w_{u, v}/(d^w (u)), W^v_u) p^u_(t). $
So if we let $W$ be the random walk matrix:
$ W^v_u = w_{u, v}/(d^w (u)), $
then $p_(t + 1) = W p_(t)$.

Alternatively, if $M$ is the weighed adjacency matrix of $G$, $D$ is the
diagonal weighted degree matrix, then we can write:
$ W = M D^(-1), $
where $D^(-1)$ is the pseudoinverse of $D$.

In Lazy Random Walks,
the walk can stay at the same vertex with some probability $p_"stay" =
1/2$.

Then, this turns the lazy random walk matrix $W$ into:
$ tilde(W) = p_"stay" I + (1 - p_"stay") M D^(-1), $
where $I$ is the identity matrix.

Now, we are interested in the eigenvalues and eigenvectors of $W$, but since $W$
is not symmetric, we cannot use the spectral decomposition directly. However, if
we let:
$ A = D^(-1/2) W D^(1/2) = D^(-1/2) M D^(-1/2), $
the normalized adjacency matrix of $G$, then $A$ is symmetric, so it has real
eigenvalues.

If $psi$ is a eigenvector of $A$ with eigenvalue $lambda$, then:
$
  A psi = lambda psi <=> D^(-1/2) W D^(1/2) psi = lambda psi <=> W (D^(1/2) psi)
  = lambda D^(1/2) psi,
$
so $lambda$ is also an eigenvalue of $W$. Hence, $A$ and $W$ has the same
spectra.

Now, note that if $d$ is the weighted degree vector, then:
$ W d = M underbrace(D^(-1) d, bold(1)) = d. $

By the Perron-Frobenius theorem, all eigenvalues of $tilde(W)$ are bounded in
$[0, 1]$. But we know $1$ is one eigenvalue, so we can denote all eigenvalues of
$tilde(W)$ by: $1 = omega_1 >= omega_2 >= ... >= omega_n >= 0$.

Denote $pi = 1/(bold(1)^T d) d$ as the normalized weighted degree vector, then
we have this following result.

#theorem[
  If the LRW starts at a node $a in V$ of a weighted graph $G = (V, E, w)$, then
  $ abs(p(t+1)^b - pi^b) <= sqrt((d(b))/(d(a))) omega_2^t, forall a in V. $
]

#proof[
  We have:
  $
    p(t) & = tilde(W)^t p(0)                                          \
         & = D^(1/2) (D^(-1/2) (p_"stay" I + (1-p_"stay")
               W) D^(1/2))^t D^(-1/2) p(0)                            \
         & = D^(1/2) (p_"stay" I + (1 - p_"stay") A)^t D^(-1/2) p(0).
  $

  Now, take $psi_1, ..., psi_n$ as eigenvectors of $A$ that forms an orthonormal
  basis, then:
  $ D^(-1/2) p(0) = sum_i underbrace((D^(-1/2) p(0))^i, c^i) psi_i $
  and
  $
    D^(1/2) (p_"stay" I + (1 - p_"stay") A)^t D^(-1/2) p(0) & = D^(1/2) sum_i c_i
                                                              (p_"stay" I + (1 - p_"stay") A)^t psi_i                   \
                                                            & = D^(1/2) sum_i c_i omega_i^t psi_i.                      \
                                                            & = D^(1/2) (c_1 psi_1 + sum_(i >= 2) c_i omega_i^t psi_i).
  $
  Since $psi_k$ are orthonormal, we have:
  $ c_i = psi_i^T D^(-1/2) p(0), forall i in ZZ^+. $
  Taking $i = 1$ gives,
  $ c_1 = psi_1 D^(-1/2) p(0). $
  Here, $psi_1 parallel d^w$, but sicne $norm(psi_1) = 1$, we must have:
  $ psi_1 = d^w/norm(d^w) . $
  Note that $D^(1/2) c_1 psi_1 = pi$, so we simply have to bound the second
  term:
  $ norm(D^(1/2) sum_(i >= 2) c_i omega_i^t psi_i)^2. $
  We know:
]

== Appendix: Pseudoinverses

Given a symmetric, positive semidefinite matrix $A$, we can do spectral
decomposition on it:
$ A = Q Lambda Q^T, "where" Q^T = Q^(-1). $
Then, we can define the pseudoinverse of $Lambda$ as:
$ (Lambda^(-1))_i^i = cases(0 "if" Lambda_i^i = 0, 1/Lambda_i^i "otherwise"). $
Then, we can define:
$ A^(-1) = (Q^T)^(-1) Lambda^(-1) Q^(-1) = Q Lambda^(-1) Q^T. $

This explains why above,
$ tr(cal(L)_G^(-1) cal(L)_G) = n - 1. $
Here, $cal(L)_G^(-1) cal(L)_G = Q Lambda^(-1) Lambda Q^T$. The product of the
two lambda does not gives $I$, but rather some $I$ with one 1 replaced by 0,
as the trace is $tr(Q Lambda^(-1) Lambda Q^T) = tr(
  Lambda Q^T Q
  Lambda^(-1)
) = tr(Lambda Lambda^(-1))$.
