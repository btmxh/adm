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

== Binary tree mechanism

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
    min (mu, nu) = min_(gamma in RR^(n times m)) <C, gamma>\
    "s.t." gamma >= 0, gamma bold(1)_m = mu, gamma^T bold(1)_n = nu.
  $
  The feasible set is called the *transportation polytope*.
] <def:ot2>

Consider a complete bipartite graph with vertex set $[n] union [m]'$.

#theorem[
  If $gamma$ is an extremal point of the transportation polytope. Then, $gamma$
  has at most $m + n - 1$ non-zero entries.
]

#proof[
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
