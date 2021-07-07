//
//  NewtonIteration.swift
//  SwiftLevMarqTest
//
//  Created by Tuukka on 2021/07/02.
//

import Foundation

/**
Does  optimizaiton by Newton iteration on the provided parameters
    
 - Parameter f: function to minimize
 - Parameter X: desired values of function
 - Parameter P: parameters to optimize
 - Parameter x: input values to function
 
 - Returns: optimized parameters
*/
func newtonIteration(f: OptFunc, X: [Double], P: [Double]) -> [Double] {
    let minTolerableError: Double = 0.00001
    let max_iters = 10_000
    var currP = P
    for i in 0..<max_iters {
        //let Xp = x.map{ f(currP, $0) }
        let Xp = f(currP)
        //let Xp = f(currP, x)
//        let error = zip(Xp, X).map { (Xpvec, Xvec) in
//            zip(Xpvec, Xvec).map { (Xpval, Xval) in
//                Xpval - Xval
//            }
//        }
        // calculate average error for each value
        var error = [Double](repeating: 0.0, count: X.count)
        
        for idx in 0..<X.count {
            error[idx] += Xp[idx] - X[idx]
        }

//        for (Xpvec, Xvec) in zip(Xp, X) {
//            for idx in 0..<Xpvec.count {
//                error[idx] += Xpvec[idx] - Xvec[idx]
//            }
//        }
        
//        for idx in 0..<error.count {
//            error[idx] /= Double(Xp.count)
//        }
        
        //let error = zip(Xp, X).map{$0 - $1}
        let errorsum = error.reduce(0.0) { r, d in
            r + (d*d)
        }
//        let errorsum = error.reduce(0.0) { rv, dvec in
//            rv + dvec.reduce(0.0) { r, d in
//                r + (d*d)
//            }
//        }
        
        //print("\(i)/\(max_iters) err: \(errorsum)")
        if errorsum < minTolerableError {
            //print("NI converged")
            break
        }
        
        let J = calculateJacobian(f: f, P: P)
        let Jinv = invert(matrix: J, M: X.count, N: P.count)
        
        let update = inner(A: Jinv, B: error, M: X.count, P: P.count, N: 1)
        let zipped = zip(currP, update)
        currP = zipped.map{$0 + (-1.0)*$1}
    }
    return currP
}
