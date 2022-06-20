$$
\begin{align}
y&=(x-mean)\frac{1}{std}*weight+bias
\\
&=(x-mean+bias*\frac{std}{weight})*\frac{weight}{std}
\\
&=(x-(mean-bias*\frac{std}{weight}))*\frac{weight}{std}
\end{align}
$$

令$new\_mean=mean-bias*\frac{std}{weight}$，$new\_std=\frac{std}{weight}$

则有
$$
y=(x-new\_mean)*\frac{1}{new\_std}
$$
