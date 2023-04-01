# wave-equation

![wave_fig](https://user-images.githubusercontent.com/87738807/229322949-bca859e1-7a9b-4329-84be-2020e7df0f7e.png)

## Phyical Description of *Waves*
The *wave equation* is a second-order partial differential equation describing *waves*, including traveling and standing waves; the latter can be considered as linear superpositions of waves traveling in opposite directions. We are going to focus on the scalar wave equation describing waves in scalars by scalar functions $u = u (x_1, x_2, ..., x_n; t)$ of a time variable $t$ (a variable representing time) and one or more spatial variables $x_1$, $x_2$, ..., $x_n$ (variables representing a position in a space under discussion) while there are vector wave equations describing waves in vectors such as waves for electrical field, magnetic field, and magnetic vector potential and elastic waves. By comparison with vector wave equations, the scalar wave equation can be seen as a special case of the vector wave equations.

The scalar wave equation can expressed as:

$$
 \nabla ^2  u \left( \vec{\mathbf{x}}, t \right) = \frac{1}{c^2} \frac{\partial^2}{\partial t^2} u\left( \vec{\mathbf{x}}, t \right)
$$

where $c$ is a fixed non-negative real coefficient representing the speed of propagation of the wave. Here

* $u$ is the factor representing a displacement from rest situation.
* $t$ represents time.
* $\partial^2 u / \partial t^2$ is a term for how the displacement accelerates.
* $\vec{\mathbf{x}}$ represents space or position.
* $\nabla ^2 u$ is a term for how the displacement is varying at the point $\vec{\mathbf{x}}$.

## Python Code
The visualization of the wave equation solution in both 1-dimensional and 2-dimensional cases is performed using Python, which provides a powerful and flexible environment for scientific computing (`numpy`) and data visualization (`matplotlib`).
