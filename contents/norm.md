{{TEX|done}}

A [[mapping]] $x\rightarrow\lVert x\rVert$ from a [[vector space]] $X$ over the field of real or complex numbers into the real numbers, subject to the conditions:
# $\lVert x\rVert\geq 0$, and $\lVert x\rVert=0$ for $x=0$ only;
# $\lVert\lambda x\rVert=\lvert\lambda\rvert\cdot\lVert x\rVert$ for every scalar $\lambda$;
# $\lVert x+y\rVert\leq\lVert x\rVert+\lVert y\rVert$ for all $x,y\in X$ (the triangle axiom).

The number $\lVert x\rVert$ is called the norm of the element $x$.

A vector space $X$ with a distinguished norm is called a [[normed space]]. A norm induces on $X$ a [[metric]] by the formula $dist(x,y)=\lVert x-y\rVert$, hence also a topology compatible with this metric. And so a normed space is endowed with the natural structure of a [[topological vector space]]. A normed space that is complete in this metric is called a [[Banach space]]. Every normed space has a Banach completion.

A topological vector space is said to be normable if its topology is compatible with some norm. Normability is equivalent to the existence of a convex bounded neighborhood of zero (a theorem of Kolmogorov, 1934).

The norm in a normed vector space $X$ is generated by an [[inner product]] (that is, $X$ is isometrically isomorphic to a [[pre-Hilbert space]]) if and only if for all $x,y\in X$,
\begin{equation}
\lVert x+y\rVert^2 + \lVert x-y\rVert^2 = 2(\lVert x\rVert^2 + \lVert y\rVert^2).
\end{equation}

Two norms $\lVert\cdot\rVert_1$ and $\lVert\cdot\rVert_2$ on one and the same vector space $X$ are called equivalent if they induce the same topology. This comes to the same thing as the existence of two constants $C_1$ and $C_2$ such that
\begin{equation}
\lVert\cdot\rVert_1 \leq C_1\lVert\cdot\rVert_2 \leq C_2\lVert\cdot\rVert_1\quad \text{for all}\; x\in X.
\end{equation}

If $X$ is complete in both norms, then their equivalence is a consequence of compatibility. Here compatibility means that the limit relations
\begin{equation}
\lVert x_n-a\rVert_1\rightarrow 0,\quad\lVert x_n-b\rVert_2\rightarrow 0.
\end{equation}
imply that $a=b$.

Not every topological vector space, even if it is assumed to be locally convex, has a continuous norm. For example, there is no continuous norm on an [[infinite product]] of straight lines with the topology of coordinate-wise convergence. The absence of a continuous norm can be an obvious obstacle to the continuous imbedding of one topological vector space in another.

If $Y$ is a closed subspace of a normed space $X$, then the [[quotient space]] $X/Y$ of cosets by $Y$ can be endowed with the norm
\begin{equation}
\lVert\tilde{x}\rVert=\inf\{\lVert x\rVert\colon x\in\tilde{x}\},
\end{equation}
under which it becomes a normed space. The norm of the image of an element $x$ under the [[quotient mapping]] $X\rightarrow X/Y$ is called the quotient norm of $x$ with respect to $Y$.

The totality $X^*$ of continuous [[linear functional]]s $\psi$ on a normed space $X$ forms a Banach space relative to the norm
\begin{equation}
\lVert\psi\rVert=\sup\{\lvert\psi(x)\rvert\colon \lVert x\rVert\leq 1\}.
\end{equation}
The norms of all functionals are attained at suitable points of the unit ball of the original space if and only if the [[Reflexive space|space is reflexive]].

The totality $L(X,Y)$ of continuous (bounded) [[linear operator]]s $A$ from a normed space $X$ into a normed space $Y$ is made into a normed space by introducing the operator norm:
\begin{equation}
\lVert A\rVert=\sup\{\lVert Ax\rVert\colon \lVert x\rVert\leq 1\}.
\end{equation}
Under this norm $L(X,Y)$ is complete if $Y$ is. When $X=Y$ is complete, the space $L(X)=L(X,X)$ with multiplication (composition) of operators becomes a [[Banach algebra]], since for the operator norm
\begin{equation}
\lVert AB\rVert \leq \lVert A\rVert\cdot\lVert B\rVert,\quad\lVert I\rVert=1,
\end{equation}
where $I$ is the identity operator (the [[unit element]] of the algebra). Other equivalent norms on $L(x)$ subject to the same condition are also interesting. Such norms are sometimes called algebraic or ringed. Algebraic norms can be obtained by renorming $X$ equivalently and taking the corresponding operator norms; however, even for $\dim X=2$ not all algebraic norms on $L(x)$ can be obtained in this manner.

A [[pre-norm]], or [[semi-norm]], on a vector space $X$ is defined as a mapping $p$ with the properties of a norm except non-degeneracy: $p(x)=0$ does not preclude that $x\neq 0$. If $\dim X<\infty$, a non-zero pre-norm $p$ on $L(x)$ subject to the condition $p(AB)\leq p(A)p(B)$ actually turns out to be a norm (since in this case $L(x)$ has no non-trivial two-sided ideals). But for infinite-dimensional normed spaces this is not so. If $X$ is a Banach algebra over $C$, then the [[spectral radius]]
\begin{equation}
\lvert x\rvert=\lim_{n\rightarrow\infty}\lVert x^n\rVert^{1/n}
\end{equation}
is a semi-norm if and only if it is uniformly continuous on $X$, and this condition is equivalent to the fact that the quotient algebra by the radical is commutative.



====Comments====

The theorem that the norms of all functionals are attained at points of the unit ball of the original space $X$ if and only if $X$ is reflexive is called James' theorem.

For norms in algebra see [[Norm on a field]] or ring (see also [[Valuation]]).

The norm of a group is the collection of group elements that commute with all subgroups, that is, the intersection of the normalizers of all subgroups (cf. [[Normalizer of a subset]]). The norm contains the [[centre of a group]] and is contained in the second [[hypercentre]] $Z_2$. For groups with a trivial centre the norm is the trivial subgroup $E$.



====References====

<table>
<TR><TD valign="top">[1]</TD> <TD valign="top">  A.N. Kolmogorov,   S.V. Fomin,   "Elements of the theory of functions and functional analysis" , '''1–2''' , Graylock  (1957–1961)  (Translated from Russian)</TD></TR>
<TR><TD valign="top">[2]</TD> <TD valign="top">  W.I. [V.I. Sobolev] Sobolew,   "Elemente der Funktionalanalysis" , H. Deutsch , Frankfurt a.M.  (1979)  (Translated from Russian)</TD></TR>
<TR><TD valign="top">[3]</TD> <TD valign="top">  G.E. Shilov,   "Mathematical analysis" , '''1–2''' , M.I.T.  (1974)  (Translated from Russian)</TD></TR>
<TR><TD valign="top">[4]</TD> <TD valign="top">  L.V. Kantorovich,   G.P. Akilov,   "Functionalanalysis in normierten Räumen" , Akademie Verlag  (1977)  (Translated from Russian)</TD></TR>
<TR><TD valign="top">[5]</TD> <TD valign="top">  W. Rudin,   "Functional analysis" , McGraw-Hill  (1979)</TD></TR>
<TR><TD valign="top">[6]</TD> <TD valign="top">  M.M. Day,   "Normed linear spaces" , Springer  (1973)</TD></TR>
<TR><TD valign="top">[7]</TD> <TD valign="top">  I.M. Glazman,   Yu.I. Lyubich,   "Finite-dimensional linear analysis: a systematic presentation in problem form" , M.I.T.  (1974)  (Translated from Russian)</TD></TR>
<TR><TD valign="top">[8]</TD> <TD valign="top">  B. Aupetit,   "Propriétés spectrales des algèbres de Banach" , Springer  (1979)</TD></TR>
<TR><TD valign="top">[9]</TD> <TD valign="top">  A.D. Grishiani,   "Theorems and problems in functional analysis" , Springer  (1982)  (Translated from Russian)</TD></TR>
<TR><TD valign="top">[10]</TD> <TD valign="top">  B. Beauzamy,   "Introduction to Banach spaces and their geometry" , North-Holland  (1982)</TD></TR>
<TR><TD valign="top">[11]</TD> <TD valign="top">  J. Lindenstrauss,   L. Tzafriri,   "Classical Banach spaces" , '''1–2''' , Springer  (1977–1979)</TD></TR>
<TR><TD valign="top">[12]</TD> <TD valign="top">  A.G. Kurosh,   "The theory of groups" , '''1–2''' , Chelsea  (1955–1956)  (Translated from Russian)</TD></TR>
<TR><TD valign="top">[13]</TD> <TD valign="top">  D.J.S. Robinson,   "Finiteness conditions and generalized solvable groups" , '''2''' , Springer  (1972)  pp. 45</TD></TR>
</table>
