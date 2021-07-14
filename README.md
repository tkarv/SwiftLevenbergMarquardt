# SwiftLevenbergMarquardt

A Levenberg-Marquardt solver implemented 100% in Swift. Details mostly
referenced from Hartley-Zisserman book.

Included are also a Gradient Descent solver and a Newton iteration based
solver.

# Usage

Two modes are provided, `optimize` and `optimizeWithInputOutput`.

`optimize` optimizes provided parameters `P` against provided output `X`.
`optimizeWithInputOutput` requires additionally sample inputs `x`, and
evaluates function performance using those.
