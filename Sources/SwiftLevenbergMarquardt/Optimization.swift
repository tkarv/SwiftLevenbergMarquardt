//
//  Optimization.swift
//  SwiftLevMarqTest
//
//  Created by Tuukka on 2021/07/01.
//

import Foundation
import Accelerate

/// Optimization function prototype
public typealias OptFunc = (_ params: [Double]) -> [Double]
public typealias OptFuncWithInputOutput = (_ params: [Double], _ x: [Double]) -> [Double]
//public typealias Optimizer = (_ f: (_ params: [Double]) -> [Double], _ X: [Double], _ P: [Double]) -> [Double]

public protocol Optimizer {
    func optimize(f: OptFunc, X: [Double], P: [Double], mask: [Double]?) -> [Double]
    func optimizeWithInputOutput(f: OptFuncWithInputOutput, X: [Double], P: [Double], x: [Double]) -> [Double]
}

/**
 Solve a system of linear equations (Ax = b)
 
 - Parameter A: matrix A
 - Parameter b: matrix b
 - Parameter AM: rows of A
 - Parameter AN: cols of A
 - Parameter bM  rows of b
 - Parameter bN: cols of b
 
 - Returns: A vector of solutions x, to the equations
 */
func solve(A: [Double], b: [Double], AM: Int, AN: Int, bM: Int, bN: Int) -> [Double] {

    var _A = A
    var _b = b
    var TRANS: CChar = 78 // 'N'
    var _AM: __CLPK_integer = __CLPK_integer(AM)
    var _AN: __CLPK_integer = __CLPK_integer(AN)
    var _bM: __CLPK_integer = __CLPK_integer(bM)
    var _bN: __CLPK_integer = __CLPK_integer(bN)

    var info: __CLPK_integer = 0
    
    var MN: Int = min(AM, AN)
    var lwork: __CLPK_integer = __CLPK_integer(MN) + __CLPK_integer(max(MN, bN))
    var workspace = [Double](repeating: 0.0, count: Int(lwork))

    withUnsafeMutablePointer(to: &TRANS) { TRANSp in
        withUnsafeMutablePointer(to: &_AM) { AMp in
            withUnsafeMutablePointer(to: &_AN) { ANp in
                withUnsafeMutablePointer(to: &_bM) { bMp in
                    withUnsafeMutablePointer(to: &_bN) { bNp in
                        dgels_(TRANSp, AMp, ANp, bNp, &_A, AMp, &_b, bMp, &workspace, &lwork, &info)
                    }
                }
            }
        }
    }
    return _b
}

/**
 Multiply double precision matrices.
 
 - Parameter A: lhs matrix (of dimensions M x P
 - Parameter B: rhs matrix (of dimensions P x N
 - Parameter M: rows of matrix A
 - Parameter P: rows of matrix B
 - Parameter N: cols of matrix B
 
 - Returns: A M x N matrix that is the result of A * B
 */
func inner(A: [Double], B: [Double], M: Int, P: Int, N: Int) -> [Double] {
//    let aStride = vDSP_Stride(1)
//    let bStride = vDSP_Stride(1)
//    let cStride = vDSP_Stride(1)
//
    var C = [Double](repeating: 0.0, count: M*N)
    
    // Using A, B, C and M, N, K as defined

    // Row-major indicates the row is contiguous in
    // memory. The other option is column-major ordering
    let Order = CblasColMajor

    // If matrix A should be transposed
    let TransposeA = CblasNoTrans

    // If matrix B should be transposed
    let TransposeB = CblasNoTrans

    // Scaling factor for A * B
    let alpha = Double(1.0)

    // Scaling factor for matrix C
    let beta = Double(1.0)

    // In row-major ordering, the number of items
    // in a row of matrix A (K)
    let lda = M

    // In row-major ordering, the number of items
    // in a row of matrix B (N)
    let ldb = P

    // In row-major ordering, the number of items
    // in a row of matrix C (N)
    let ldc = M

    cblas_dgemm(
        Order,
        TransposeA, TransposeB,
        Int32(M), Int32(N), Int32(P),
        alpha,
        A, Int32(lda),
        B, Int32(ldb),
        beta,
        &C, Int32(ldc)
    )
    
//    vDSP_mmulD(
//        A, aStride,
//        B, bStride,
//        &C, cStride,
//        vDSP_Length(M),
//        vDSP_Length(N),
//        vDSP_Length(P)
//    )
    return C
}

/**
 Transpose matrix
 
 - Parameter A: M-by-N matrix
 - Parameter M: number of rows in A
 - Parameter N: number of cols in A
 
 - Returns: N-by-M matrix that is the transpose of A
 */
func transpose(A: [Double], M: Int, N: Int) -> [Double] {
    var ret: [Double] = []
    for col in 0..<M {
        for row in 0..<N {
            ret.append(A[row*M+col])
        }
    }
    return ret
}

/**
 Invert matrix
 
 - Parameter matrix: M-by-N matrix
 - Parameter M: number of rows in matrix
 - Parameter N: number of columns in matrix
 */
func invert(matrix : [Double], M: Int, N: Int) -> [Double] {
    var inMatrix = matrix
    var _M: __CLPK_integer = __CLPK_integer(M)
    var _N: __CLPK_integer = __CLPK_integer(N)
    // var N = __CLPK_integer(sqrt(Double(matrix.count)))
    var smallerDim = min(_M, _N)
    var pivots = [__CLPK_integer](repeating: 0, count: Int(smallerDim))
    var workspace = [Double](repeating: 0.0, count: Int(N))
    var error : __CLPK_integer = 0
    
    withUnsafeMutablePointer(to: &_M, { ptrToM in
        withUnsafeMutablePointer(to: &_N) { ptrToN in
            withUnsafeMutablePointer(to: &smallerDim) { ptrToSmallerDim in
                var ret = dgetrf_(ptrToM, ptrToN, &inMatrix, ptrToM, &pivots, &error)
                guard error == 0 else {
                    print("matrix inverse failed LU factorization with INFO code: \(error)")
                    print("""
                        INFO    (output) INTEGER
                                = 0:  successful exit
                                < 0:  if INFO = -i, the i-th argument had an illegal
                                value
                                
                                > 0:  if INFO = i, U(i,i) is exactly zero. The fac-
                                torization has been completed, but the factor U is
                                exactly singular, and division by zero will occur if
                                it is used to solve a system of equations.
                        """)
                    return
                }
                ret = dgetri_(ptrToN, &inMatrix, ptrToM, &pivots, &workspace, ptrToN, &error)
                guard error == 0 else {
                    print("matrix inverse failed inversion with INFO code: \(error)")
                    print("""
                        INFO    (output) INTEGER
                                = 0:  successful exit
                                < 0:  if INFO = -i, the i-th argument had an illegal
                                value
                                > 0:  if INFO = i, U(i,i) is exactly zero; the
                                matrix is singular and its inverse could not be com-
                                puted.
                        """)
                    return
                }
            }
        }
    })
    
    return inMatrix
}

/**
 Compute delta for numerical approximation.
 Delta is computed as max(1e-6, 1e-4 * x_i for x_i in X)
 
 Ref: Hartley-Zisserman p. 602
 
 - Parameter X: values provided for optimization
 */
func computeDelta(X: Double) -> Double {
    return max(1e-6, abs(X) * 1e-4)
}

///**
// Calculate Jacobian of function f with respect to parameters P.
// We calculate numerically using the provided values X
// */
//func calculateJacobian(f: OptFunc, P: [Double]) -> [Double] {
//    //let d = computeDelta(X: X)
//
//    var J: [Double] = []
//
//    // calculate original value
//    let y0 = f(P)
//
//    for (n, p) in P.enumerated() {
//        // calculate jacobian one parameter at a time
//        // end result is a flattened M by N matrix where elements at (m,n) are
//        // ∂f_m / ∂p_n
//        // elements are in column-major order
//        // (calculated for one parameter delta at a time)
//        var newP = P
//
//        let d = computeDelta(X: p)
//
//        newP[n] += d
//
//        let y1 = f(newP)
//
//        let grad = zip(y1, y0).map { y1v, y0v in
//            (y1v - y0v) / d
//        }
//
//        J.append(contentsOf: grad)
//    }
//
//    return J // transpose(A: J, M: P.count, N: y0.count)
//}

/**
 Calculate Jacobian of function f with respect to parameters P.
 We calculate numerically using the provided values X
 Mask declares which values to ignore gradient for
 */
func calculateJacobian(f: OptFunc, P: [Double], mask: [Double]? = nil) -> [Double] {
    //let d = computeDelta(X: X)
    
    var J: [Double] = []
    
    // calculate original value
    let y0 = f(P)
    
    for (n, p) in P.enumerated() {
        // calculate jacobian one parameter at a time
        // end result is a flattened M by N matrix where elements at (m,n) are
        // ∂f_m / ∂p_n
        // elements are in column-major order
        // (calculated for one parameter delta at a time)
        var newP = P
        
        let d = computeDelta(X: p)
        
        newP[n] += d
        
        let y1 = f(newP)
        
        var grad: [Double] = []
        if mask == nil {
            grad = zip(y1, y0).map { y1v, y0v in
                (y1v - y0v) / d
            }
        } else {
            for idx in 0..<mask!.count {
                let y0v = y0[idx]
                let y1v = y1[idx]
                let mval = mask![idx]
                
                grad.append(
                    ((y1v - y0v) / d) * mval
                )
            }
        }
        
        J.append(contentsOf: grad)
    }
    
    return J // transpose(A: J, M: P.count, N: y0.count)
}

/**
 Calculate Jacobian of function f with respect to parameters P.
 We calculate numerically using the provided values X
 */
func calculateJacobian(f: OptFuncWithInputOutput, P: [Double], x: [Double]) -> [Double] {
    //let d = computeDelta(X: X)
    
    var J: [Double] = []
    
    // calculate original value
    let y0 = f(P, x)
    
    for (n, p) in P.enumerated() {
        // calculate jacobian one parameter at a time
        // end result is a flattened M by N matrix where elements at (m,n) are
        // ∂f_m / ∂p_n
        // elements are in column-major order
        // (calculated for one parameter delta at a time)
        var newP = P
        
        let d = computeDelta(X: p)
        
        newP[n] += d
        
        let y1 = f(newP, x)
        
        let grad = zip(y1, y0).map { y1v, y0v in
            (y1v - y0v) / d
        }
        
        J.append(contentsOf: grad)
    }
    
    return J // transpose(A: J, M: P.count, N: y0.count)
}
