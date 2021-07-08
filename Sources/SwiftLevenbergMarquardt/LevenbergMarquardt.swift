import Foundation
import simd

let maxLMIterations = 10

/**
Does Levenberg-Marquardt optimizaiton on the provide dparameters
    
- Parameter f: function to minimize
- Parameter X: desired values of function
- Parameter P: parameters to optimize
 
 - Returns: optimized parameters
*/
public class LevenbergMarquardtOptimizer : Optimizer {
    public var tolerance: Double = 0.01
    public var max_iters: Int = 10_000
    
    public init() {
        
    }
    
    public func optimize(f: OptFunc, X: [Double], P: [Double]) -> [Double] {
        var currP = P
        
        //var Xp = x.map{ f(currP, $0) }
        var Xp = f(currP)
        // calculate average error for each value
        var error = [Double](repeating: 0.0, count: X.count)

        for idx in 0..<error.count {
            error[idx] += Xp[idx] - X[idx]
        }
            
        var errorsum = error.reduce(0.0) { r, d in
            r + (d*d)
        }
        
        var lastl = 0.0
        
        var iter = 0
        while iter < max_iters {
            let J = calculateJacobian(f: f, P: currP)
            
            let Jt = transpose(A: J, M: X.count, N: P.count)

            let JtJ = inner(A: Jt, B: J, M: P.count, P: X.count, N: P.count)
            
            let minusJterror = inner(A: Jt, B: error, M: P.count, P: X.count, N: 1).map{$0 * -1.0}

            if lastl == 0.0 {
                for j in 0..<P.count {
                    lastl += JtJ[j*P.count + j]
                }
                lastl /= Double(P.count)
                lastl *= 1e-3
            }
            
            var lambda: [Double] = [Double](repeating: 0.0, count: P.count * P.count)
            for j in 0..<P.count {
                lambda[j*P.count + j] = lastl * JtJ[j*P.count + j]
            }
            
            let sumelement = zip(JtJ, lambda).map{$0 + $1}

            let delta = solve(A: sumelement, b: minusJterror, AM: P.count, AN: P.count, bM: P.count, bN: 1)
            
            currP = zip(currP, delta).map{$0 + $1}
            
            Xp = f(currP) //x.map{ f(currP, $0) }
            
            let lasterrorsum = errorsum
            
            error = [Double](repeating: 0.0, count: X.count)
            
            for idx in 0..<error.count {
                error[idx] += Xp[idx] - X[idx]
            }
            
            errorsum = error.reduce(0.0) { r, d in
                r + (d*d)
            }
            
            if errorsum < lasterrorsum {
                // accept
                lastl /= 10
                iter += 1
            } else {
                for _ in 0..<maxLMIterations {
                    lastl *= 10
                    let J = calculateJacobian(f: f, P: currP)
                    
                    let Jt = transpose(A: J, M: X.count, N: P.count)

                    let JtJ = inner(A: Jt, B: J, M: P.count, P: X.count, N: P.count)
                    
                    let minusJterror = inner(A: Jt, B: error, M: P.count, P: X.count, N: 1).map{$0 * -1.0}

                    if lastl == 0.0 {
                        for j in 0..<P.count {
                            lastl += JtJ[j*P.count + P.count]
                        }
                        lastl /= Double(P.count)
                        lastl *= 1e-3
                    }
                    
                    var lambda: [Double] = [Double](repeating: 0.0, count: P.count * P.count)
                    for j in 0..<P.count {
                        lambda[j*P.count + j] = lastl * JtJ[j*P.count + j]
                    }
                    
                    let sumelement = zip(JtJ, lambda).map{$0 + $1}

                    let delta = solve(A: sumelement, b: minusJterror, AM: P.count, AN: P.count, bM: P.count, bN: 1)
                    
                    currP = zip(currP, delta).map{$0 + $1}
                    
                    Xp = f(currP)
                    
                    let lasterrorsum = errorsum
                    
                    error = [Double](repeating: 0.0, count: X.count)
                    
                    for idx in 0..<error.count {
                        error[idx] += Xp[idx] - X[idx]
                    }

                    for idx in 0..<error.count {
                        error[idx] /= Double(Xp.count)
                    }
                    
                    errorsum = error.reduce(0.0) { r, d in
                        r + (d*d)
                    }
                    iter += 1
                    if errorsum < lasterrorsum {
                        break
                    }
                }
                // solve again
            }
            print("\(iter)/\(max_iters) err: \(errorsum)")
            if errorsum < tolerance {
                //print("NI converged")
                break
            }
        }
        return currP
    }
}
