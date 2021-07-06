//
//  GradientDescent.swift
//  SwiftLevMarqTest
//
//  Created by Tuukka on 2021/07/01.
//

import Foundation
import simd
import Accelerate

/**
Does gradient descent  optimizaiton on the provide dparameters
    
 - Parameter f: function to minimize
 - Parameter X: desired output values of function
 - Parameter P: parameters to optimize
 - Parameter x: input values to function
 - Parameter r: rate of gradient descent
 
 - Returns: optimized parameters
*/
func gradientDescent(f: OptFunc, X: [[Double]], P: [Double], x: [[Double]], r: Double) -> [Double] {
    let minTolerableError: Double = 0.00001
    let max_iters = 10_000
    var currP = P
    for i in 0..<max_iters {
        let Xp = x.map{ f(currP, $0) }

        // calculate average error for each value
        var error = [Double](repeating: 0.0, count: x[0].count)
        for (Xpvec, Xvec) in zip(Xp, X) {
            for idx in 0..<Xpvec.count {
                error[idx] += Xpvec[idx] - Xvec[idx]
            }
        }
        
        for idx in 0..<error.count {
            error[idx] /= Double(Xp.count)
        }
        
        let errorsum = error.reduce(0.0) { r, d in
            r + (d*d)
        }
        
        //print("\(i)/\(max_iters) err: \(errorsum)")
        if errorsum < minTolerableError {
            //print("NI converged")
            break
        }
        
        let J = calculateJacobian(f: f, P: currP, X: x)
        
        let Jt = transpose(A: J, M: x[0].count, N: P.count)

        let JtJ = inner(A: Jt, B: J, M: P.count, P: x[0].count, N: P.count)
        
        let minusJterror = inner(A: Jt, B: error, M: P.count, P: x[0].count, N: 1).map{$0 * -1.0}
        
        var lambda: [Double] = [Double](repeating: 0.0, count: P.count * P.count)
        for j in 0..<P.count {
            lambda[j*P.count + j] = 1.0/r * JtJ[j*P.count + j]
        }

        let delta = solve(A: lambda, b: minusJterror, AM: P.count, AN: x[0].count, bM: P.count, bN: 1)
        
        currP = zip(currP, delta).map{$0 + $1}
    }
    return currP
}
