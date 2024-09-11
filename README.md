# Système solaire (simulations numériques en Python)

[vidéo d'illustration](https://www.youtube.com/watch?v=oL14Y5qqagk)

## Description du modèle testé et objectif suivi

Chacun des $N$ astres inclus dans le modèle a une position $\vec{p_n}$ et une vitesse $\vec{v_n}$ dans le référentiel inertiel [ICRS](https://fr.wikipedia.org/wiki/Syst%C3%A8me_de_r%C3%A9f%C3%A9rence_c%C3%A9leste_international).

L'équation d'état qui régit le mouvement de ces astres fait aussi intervenir leur [paramètre gravitationnel standard](https://fr.wikipedia.org/wiki/Param%C3%A8tre_gravitationnel_standard) noté $\mu_k$ :

$$
\begin{align}
\frac{d\vec{p_n}}{dt}&=\vec{v_n}\\
\frac{d\vec{v_n}}{dt}&=\sum_{1\leq k \leq N,k \neq n}-\mu_k\frac{\vec{p_n}-\vec{p_k}}{\left\lVert\vec{p_n}-\vec{p_k}\right\lVert^3}
\end{align}
$$

Ce modèle fait donc appel à la mécanique classique et aux lois de Newton pour la gravitation. Les valeurs des paramètres gravitationnels standards $\mu_k$ sont repris de l'article ["The JPL Planetary and Lunar Ephemerides DE440 and DE441"](https://iopscience.iop.org/article/10.3847/1538-3881/abd414/pdf) (Table 2, page 5).

Le programme de simulation qui résout numériquement l'équation d'état proposée permet de réaliser ensuite des comparaisons avec les [éphémérides fournies par le JPL](https://ssd.jpl.nasa.gov/horizons/app.html#/). Les conditions initiales pour les simulations numériques correspondent à la première ligne des fichiers importés du site du JPL, donc aux positions et vitesses des astres à la date de début choisie. Le pas de calcul pour les simulations numériques est le même que celui choisi pour les fichiers importés, afin de faciliter les comparaisons.

L'objectif est de montrer que la modélisation simplifiée qui est proposée donne cependant des résultats qui restent cohérents avec ces éphémérides très précises du JPL.
